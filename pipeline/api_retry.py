"""
api_retry.py — automatic retry wrapper for Anthropic API calls.

WHY THIS EXISTS
---------------
The Anthropic API occasionally returns transient 5xx server errors or
drops the network connection mid-request. When we're doing an eval run
that makes hundreds or thousands of API calls, a single such blip will
crash the whole pipeline and we lose work (or at least have to restart
from the last saved chunk).

This module wraps client.messages.create(...) with an automatic
retry-with-exponential-backoff so that transient failures get
re-attempted silently instead of crashing the run. We retry on:

  * anthropic.InternalServerError    (HTTP 5xx — Anthropic's problem)
  * anthropic.APIConnectionError     (dropped TCP, DNS hiccup, etc.)
  * anthropic.APITimeoutError        (request didn't respond in time)
  * anthropic.RateLimitError         (HTTP 429 — too many requests)
  * Other anthropic.APIStatusError   (only if status_code >= 500)

We do NOT retry on client-side errors — these are "our fault" and
won't magically fix themselves by asking again:

  * anthropic.BadRequestError         (HTTP 400 — malformed request)
  * anthropic.AuthenticationError     (HTTP 401 — bad API key)
  * anthropic.PermissionDeniedError   (HTTP 403 — not allowed)
  * anthropic.NotFoundError           (HTTP 404)
  * anthropic.UnprocessableEntityError (HTTP 422 — semantic error)
  * anthropic.ConflictError           (HTTP 409)

BACKOFF STRATEGY
----------------
Exponential backoff with a base of 2 seconds:
    attempt 1 fails -> wait  2s, try again
    attempt 2 fails -> wait  4s, try again
    attempt 3 fails -> wait  8s, try again
    attempt 4 fails -> wait 16s, try again
    attempt 5 fails -> wait 32s, try again (final attempt = 6)
    attempt 6 fails -> re-raise the exception

Rate limit (429) errors get DOUBLE the base delay because rate limits
typically clear on a per-minute window, so waiting a little longer is
much more likely to succeed.

USAGE
-----
Drop-in replacement for client.messages.create(...):

    from pipeline.api_retry import call_with_retry

    # Before:
    # response = client.messages.create(model=..., max_tokens=..., ...)

    # After:
    response = call_with_retry(client, model=..., max_tokens=..., ...)

All keyword arguments are passed through unchanged.
"""

import time
import anthropic


# Tunable constants. Keeping them at module level so tests can monkey-
# patch them if we ever want to run a fast unit test without real delays.
MAX_RETRIES = 5                # max retries after the initial attempt
BASE_DELAY_SECONDS = 2.0       # 2s, 4s, 8s, 16s, 32s
RATE_LIMIT_MULTIPLIER = 2.0    # rate limit waits 2x as long


def _is_retryable_status_error(exc):
    """
    Decide whether an APIStatusError subclass should be retried.

    We retry on 5xx server errors (Anthropic's side) but not on 4xx
    client errors (our side, won't fix with a retry).

    Uses getattr with a default because older SDK versions may not
    expose status_code on every subclass, and we prefer "retry once
    and give up" over "crash because the attribute is missing".
    """
    # InternalServerError is explicitly a 5xx — retry it
    if isinstance(exc, anthropic.InternalServerError):
        return True
    # RateLimitError is 429 but we handle it in its own branch
    if isinstance(exc, anthropic.RateLimitError):
        return True
    # Any other APIStatusError: only retry if status is 5xx
    code = getattr(exc, "status_code", None)
    if isinstance(code, int) and code >= 500:
        return True
    return False


def call_with_retry(client, **kwargs):
    """
    Call client.messages.create(**kwargs) with automatic retry on
    transient failures. Returns the response object on success.

    Prints a terminal-visible notice each time a retry occurs so the
    user running the pipeline can see that the system is self-healing
    rather than hanging silently.

    Raises the final exception if all retries are exhausted, or
    immediately if the exception type indicates a non-retryable
    client-side error.
    """
    last_exc = None
    # We do MAX_RETRIES + 1 total attempts (1 initial + MAX_RETRIES retries)
    for attempt in range(MAX_RETRIES + 1):
        try:
            return client.messages.create(**kwargs)

        except (anthropic.APIConnectionError, anthropic.APITimeoutError) as e:
            # Network/timeout — always retryable
            last_exc = e
            if attempt >= MAX_RETRIES:
                print(
                    f"    [RETRY] {type(e).__name__} after {MAX_RETRIES + 1} attempts — giving up.\n"
                    f"    [RETRY] Final error: {str(e)[:200]}"
                )
                raise
            delay = BASE_DELAY_SECONDS * (2 ** attempt)
            print(
                f"    [RETRY] Network issue ({type(e).__name__}) on attempt {attempt + 1}/"
                f"{MAX_RETRIES + 1}. Waiting {delay:.0f}s before retry. "
                f"Error: {str(e)[:120]}"
            )
            time.sleep(delay)

        except anthropic.RateLimitError as e:
            # 429 — wait a bit longer than server errors
            last_exc = e
            if attempt >= MAX_RETRIES:
                print(
                    f"    [RETRY] Rate limited after {MAX_RETRIES + 1} attempts — giving up.\n"
                    f"    [RETRY] Final error: {str(e)[:200]}"
                )
                raise
            delay = BASE_DELAY_SECONDS * (2 ** attempt) * RATE_LIMIT_MULTIPLIER
            print(
                f"    [RETRY] Rate limited (attempt {attempt + 1}/{MAX_RETRIES + 1}). "
                f"Waiting {delay:.0f}s before retry."
            )
            time.sleep(delay)

        except anthropic.APIStatusError as e:
            # Catches InternalServerError (5xx) and all other HTTP
            # status errors. Decide whether this specific one is worth
            # retrying (5xx yes, 4xx no).
            last_exc = e
            if not _is_retryable_status_error(e):
                # 4xx client error — re-raise immediately, no point retrying
                print(
                    f"    [RETRY] Non-retryable API error ({type(e).__name__}): {str(e)[:200]}"
                )
                raise
            if attempt >= MAX_RETRIES:
                code = getattr(e, "status_code", "?")
                print(
                    f"    [RETRY] Server error {code} ({type(e).__name__}) after "
                    f"{MAX_RETRIES + 1} attempts — giving up.\n"
                    f"    [RETRY] Final error: {str(e)[:200]}"
                )
                raise
            code = getattr(e, "status_code", "?")
            delay = BASE_DELAY_SECONDS * (2 ** attempt)
            print(
                f"    [RETRY] Server error {code} ({type(e).__name__}) on attempt "
                f"{attempt + 1}/{MAX_RETRIES + 1}. Waiting {delay:.0f}s before retry. "
                f"Error: {str(e)[:120]}"
            )
            time.sleep(delay)

    # This point should be unreachable — loop either returns success
    # or re-raises. But if somehow we fall through, raise what we have.
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("call_with_retry exhausted loop without success or exception")
