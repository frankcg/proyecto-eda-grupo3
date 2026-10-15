"""Etapa 0 (S1) - Cola FIFO implementada con lista enlazada simple.

Operaciones:
    enqueue  O(1)   - inserta al final (tail)
    dequeue  O(1)   - extrae del frente (head)
    dequeue_batch(b) O(b)
Memoria: O(n) nodos. A diferencia de ``list.pop(0)`` (O(n)) no desplaza
elementos, y a diferencia de un arreglo circular no requiere capacidad fija,
lo que la hace adecuada como buffer de un stream de tamaño desconocido.
"""
from __future__ import annotations


class _Node:
    __slots__ = ("value", "next")

    def __init__(self, value):
        self.value = value
        self.next = None


class LinkedQueue:
    """Cola FIFO con lista enlazada y capacidad máxima opcional (backpressure)."""

    __slots__ = ("_head", "_tail", "_size", "capacity", "enqueued", "dropped")

    def __init__(self, capacity: int | None = None):
        self._head = None
        self._tail = None
        self._size = 0
        self.capacity = capacity
        self.enqueued = 0      # métricas para el dashboard
        self.dropped = 0

    def __len__(self) -> int:
        return self._size

    def is_empty(self) -> bool:
        return self._size == 0

    def enqueue(self, value) -> bool:
        """Encola un elemento. Devuelve False si la cola está llena (backpressure)."""
        if self.capacity is not None and self._size >= self.capacity:
            self.dropped += 1
            return False
        node = _Node(value)
        if self._tail is None:
            self._head = self._tail = node
        else:
            self._tail.next = node
            self._tail = node
        self._size += 1
        self.enqueued += 1
        return True

    def dequeue(self):
        if self._head is None:
            raise IndexError("dequeue de una cola vacía")
        node = self._head
        self._head = node.next
        if self._head is None:
            self._tail = None
        self._size -= 1
        return node.value

    def peek(self):
        if self._head is None:
            raise IndexError("peek de una cola vacía")
        return self._head.value

    def dequeue_batch(self, b: int) -> list:
        """Extrae hasta ``b`` elementos en orden FIFO (micro-lote)."""
        out = []
        node = self._head
        while node is not None and len(out) < b:
            out.append(node.value)
            node = node.next
        self._head = node
        if node is None:
            self._tail = None
        self._size -= len(out)
        return out

    def __iter__(self):
        node = self._head
        while node is not None:
            yield node.value
            node = node.next
