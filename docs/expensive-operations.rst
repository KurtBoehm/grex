.. cpp:namespace:: grex

.. _expensive-operations:

####################
Expensive Operations
####################

Some backend operations can be realized by many different instruction sequences, and which of them is available — let alone fastest — depends on a compile-time parameter such as a blend pattern or a shuffle index list.
Crucially, some variants of an operation may be composed of multiple steps while some may not (e.g., shuffling with selective zeroing, which some shuffle instructions support and some do not), making a static pecking order very hard to determine.

Grex therefore models these operations as a set of independent **candidates** annotated with a cost, and lets a small compile-time framework pick the cheapest applicable one.
This page describes that framework; the operations built on it are :doc:`operations/blend-static` and :doc:`operations/shuffle-static`.

Everything here is internal to the backend.
It is documented because it explains the structure of those implementations, not because it is part of the public interface.

.. _expensive-operations-candidates:

*******************
Candidate Interface
*******************

A candidate is an empty type deriving from ``BaseExpensiveOp``, which the ``AnyExpensiveOp`` concept detects.
Each candidate provides three static members, all parameterized by the compile-time value being dispatched on (passed as an ``AutoTag``):

.. list-table::
   :header-rows: 1
   :widths: 1 3

   * - Member
     - Purpose
   * - ``is_applicable(tag)``
     - ``constexpr bool``: whether this candidate can realize the requested pattern at all.
       A candidate that needs an instruction from a higher x86-64 level, or whose granularity does not match the pattern, reports ``false``.
   * - ``apply(args…, tag)``
     - The actual instruction sequence.
       Only ever instantiated for the winning candidate.
   * - ``cost(tag)``
     - ``constexpr Cost``: the estimated cost of ``apply``.

Because ``is_applicable`` and ``cost`` are ``constexpr`` and ``apply`` is only instantiated once the winner is known, an unselected candidate never contributes any code and never needs its intrinsics to exist on the current target.

Composite candidates compute their cost from the candidates they delegate to.
A candidate that splits a super-native vector into halves, for instance, sums the inverse throughputs and latencies of the two half-sized operations, so the comparison at the top level accounts for the whole expansion rather than just its outermost step.

.. _expensive-operations-cost:

****
Cost
****

.. cpp:struct:: backend::Cost

   .. cpp:member:: f64 inv_throughput

      Inverse throughput: how much of the relevant execution port the sequence occupies, so lower is better.

   .. cpp:member:: f64 latency

      Latency of the dependency chain through the sequence.

   The comparison operator is defaulted, so costs compare lexicographically in declaration order: inverse throughput first, latency only as a tie-breaker.
   This deliberately favours sequences that occupy fewer execution slots even when they take slightly longer.

   A cost of ``{0, 0}`` marks a candidate that emits nothing at all — for example a blend whose selectors happen to request exactly one of its inputs.

.. _expensive-operations-selection:

*********
Selection
*********

Two aliases turn a candidate list into a decision:

- ``ApplicableTypes<value, Candidates…>`` filters the list to those candidates whose ``is_applicable`` accepts ``value``, preserving the original order in a ``TypeSeq``.
- ``CheapestType<value, Candidates…>`` applies that filter and then reduces the survivors to the one with the lowest ``cost``.

The reduction keeps the earlier candidate when two costs compare equal, so **the order of a candidate list is its tie-break order**.

``CheapestType`` has no case for an empty candidate list, so a list with no applicable candidate is a compile error.
Every list therefore ends with a candidate whose ``is_applicable`` is unconditionally ``true``:

.. list-table::
   :header-rows: 1

   * - Operation
     - Universal fallback
     - Strategy
   * - :ref:`Compile-time blend <operations-blend-static>`
     - ``BlenderVariable``
     - Materialize a mask from the selectors and use the run-time :cpp:func:`~backend::blend`.
   * - :ref:`Compile-time blend_zero <operations-blend-zero-static>`
     - ``ZeroBlenderAnd``
     - Bitwise AND with a constant mask vector.
   * - :ref:`Compile-time shuffle <operations-shuffle-static>`
     - varies by register width
     - ``ShufflerExtractSet`` (128-bit and Neon), ``ShufflerShuffle8x32Ext`` (256-bit), or a ``permutexvar``-based candidate (512-bit).

The shuffle row illustrates that the fallback is whatever happens to cover every remaining pattern most cheaply at a given width, not a fixed strategy: the generic extract-and-rebuild sequence is only worth keeping where no single wide permute subsumes it.

Since all of this resolves during template instantiation, the generated code contains only the winning sequence; the candidate list, the applicability tests, and the cost comparison leave no trace.

.. _expensive-operations-dispatch:

*************
Size Dispatch
*************

Cost-based selection only decides *how* to implement an operation on a register that the hardware actually has.
Choosing between native, sub-native, and super-native handling happens one level up, through partial specializations of the dispatch trait constrained on the total pattern width:

- Narrower than the smallest native register: the sub-native candidate, which pads the pattern to the full backing register and defers to the native decision.
- Wider than the largest native register: the super-native candidate, which splits the pattern into halves and defers to two narrower decisions.
- Otherwise: a ``CheapestType`` over the candidate list for that register width and backend.

These sub- and super-native candidates are selected unconditionally by size, not by cost, even though they implement the same candidate interface so that they can report a composite cost to whatever encloses them.
