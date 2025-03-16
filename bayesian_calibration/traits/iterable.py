from typing import Protocol, Iterator


class Iterable(Protocol):
    def __iter__(self) -> Iterator: ...

    def __len__(self) -> int: ...
