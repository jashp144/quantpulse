"""FIFO event queue mechanism for QuantPulse event loop."""

from collections import deque

from quantpulse.engine.events import Event


class EventQueue:
    """Deterministic chronological FIFO event bus for backtesting."""

    def __init__(self) -> None:
        self._queue: deque[Event] = deque()

    def put(self, event: Event) -> None:
        """Push an event to the tail of the event queue."""
        self._queue.append(event)

    def get(self) -> Event | None:
        """Pop the oldest event from the head of the event queue."""
        if self._queue:
            return self._queue.popleft()
        return None

    def empty(self) -> bool:
        """Check if queue is empty."""
        return len(self._queue) == 0

    def qsize(self) -> int:
        """Current number of pending events."""
        return len(self._queue)

    def clear(self) -> None:
        """Empty the event queue."""
        self._queue.clear()
