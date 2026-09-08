/**
 * Loading / error / empty states.
 *
 * Pulled into one file because this app has an unusual number of legitimate
 * empty states -- no model yet, no features for a game yet, no games ingested
 * yet -- and each needs to explain ITSELF rather than showing a generic
 * "no data". A user seeing a blank prediction panel should learn that the
 * model hasn't been trained, not wonder whether the page is broken.
 */

export function Loading({ rows = 6 }: { rows?: number }) {
  return (
    <div className="game-grid" aria-busy="true" aria-label="Loading games">
      {Array.from({ length: rows }, (_, i) => (
        <div className="skeleton" key={i} />
      ))}
    </div>
  );
}

export function ErrorState({ message }: { message: string }) {
  return (
    <div className="card state error" role="alert">
      <div className="title">Something went wrong</div>
      <p className="detail">{message}</p>
    </div>
  );
}

export function EmptyState({ title, detail }: { title: string; detail: string }) {
  return (
    <div className="card state">
      <div className="title">{title}</div>
      <p className="detail">{detail}</p>
    </div>
  );
}

/** A smaller empty state that sits inside a panel rather than replacing the page. */
export function InlineEmpty({ title, detail }: { title: string; detail: string }) {
  return (
    <div className="empty-inline">
      <strong>{title}</strong>
      {detail}
    </div>
  );
}
