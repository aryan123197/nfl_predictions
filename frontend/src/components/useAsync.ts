import { useEffect, useState } from "react";
import { ApiError } from "../api";

export interface AsyncState<T> {
  data: T | null;
  error: string | null;
  /** Set when the failure was an HTTP status, so callers can treat 404/501
   *  as a legitimate empty state rather than an error. */
  status: number | null;
  loading: boolean;
}

/**
 * Minimal data-fetching hook.
 *
 * Guards against the two classic bugs rather than pulling in a query
 * library for three endpoints:
 *   - a resolved request from a previous key overwriting a newer one
 *     (stale response race), and
 *   - a setState after unmount.
 * Both are handled by the `cancelled` flag.
 */
export function useAsync<T>(fn: () => Promise<T>, deps: unknown[]): AsyncState<T> {
  const [state, setState] = useState<AsyncState<T>>({
    data: null,
    error: null,
    status: null,
    loading: true,
  });

  useEffect(() => {
    let cancelled = false;
    setState({ data: null, error: null, status: null, loading: true });

    fn()
      .then((data) => {
        if (!cancelled) setState({ data, error: null, status: null, loading: false });
      })
      .catch((err: unknown) => {
        if (cancelled) return;
        const message = err instanceof Error ? err.message : String(err);
        const status = err instanceof ApiError ? err.status : null;
        setState({ data: null, error: message, status, loading: false });
      });

    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);

  return state;
}
