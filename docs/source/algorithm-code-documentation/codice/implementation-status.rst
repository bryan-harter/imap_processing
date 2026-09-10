.. _codice-implementation-status:

Implementation Status and Known Gaps
====================================

This page is the honest accounting of where the code stands against the
algorithm document. **Read it before proposing or estimating work.**

Accurate as of the most recent survey of ``imap_processing/codice`` and
``imap_processing/ialirt/l0/process_codice.py``, against algorithm document
Rev 3 Chg 0. If you change something material, update this page in the same
commit.

Summary
-------

.. list-table::
   :header-rows: 1
   :widths: 14 22 64

   * - Level
     - State
     - Notes
   * - L1A
     - **Mostly complete**
     - Eleven of fourteen science products implemented, plus housekeeping. The
       SCI-LUT unpacking machinery, all seven compression modes, segmented
       direct events and the full P3 NSO masking rules all work. **The three
       angular/NSW-species products are missing entirely.**
   * - L1B
     - **Complete for what L1A produces**
     - Every implemented L1A product has a working rate conversion. The three
       missing L1A products would crash if fed in.
   * - L2
     - **Roughly half**
     - Five of eight declared products implemented: ``lo-sw-species``,
       ``lo-direct-events``, ``hi-omni``, ``hi-sectored``,
       ``hi-direct-events``. The three Lo angular/NSW products are not built.
       The pass-through branch for counters/priority products is broken.
   * - L3
     - **Out of scope**
     - Belongs to a different repository. See :ref:`codice-l3-scope`.
   * - I-ALiRT
     - **Works end to end**
     - Lo abundance/charge-state ratios and Hi H intensities both produced.
       Uncertainty propagation is missing.

There are **no** ``NotImplementedError`` raises anywhere in the CoDICE code,
apart from the generic unknown-data-level branch in ``Codice.do_processing``.

Not implemented at all
----------------------

Lo angular counts and intensities
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**The largest single gap.** Document sections 10.3.4 (L1A), 11.2.3 (L1B) and
12.2.2 (L2).

* APIDs 1158 (``COD_LO_SW_ANGULAR_COUNTS``) and 1159
  (``COD_LO_NSW_ANGULAR_COUNTS``) are defined in ``CODICEAPID`` but
  **``process_l1a`` has no branch for them**. There is no
  ``codice_l1a_lo_angular.py``.
* ``LO_SW_ANGULAR_VARIABLE_NAMES`` and ``LO_NSW_ANGULAR_VARIABLE_NAMES`` exist
  in ``constants.py`` and are otherwise unreferenced.
* ``imap_codice_l1a_lo-sw-angular``, ``imap_codice_l1b_lo-sw-angular``,
  ``imap_codice_l2_lo-sw-angular`` and the NSW equivalents all have
  ``Logical_source`` entries and CDF variable-attribute YAML
  (``imap_codice_l2-lo-angular_variable_attrs.yaml``), so the *metadata* is
  ready and the *processing* is not.

Building it requires, in order:

1. The full **de-spin** mapping from (12 spin sectors x 5/19 positions) to
   (24 spin angles x 5/19 positions), using half-spin parity and pixel
   orientation. See :ref:`codice-esa-stepping`.
2. The P3 **NSO masking** rules (already written twice, in
   ``codice_l1a_lo_priority.py`` and ``codice_l1a_lo_counters_singles.py`` -
   worth factoring out rather than writing a third time).
3. L1B: ``n_sectors = 1``. Trivial once ``LO_SW_ANGULAR_VARIABLE_NAMES`` is
   reachable by the ``getattr`` reflection.
4. L2: the intensity division, then the **position -> elevation angle**
   reduction, then the **position 1 and 13 replication** across the unobserved
   half of the spin angles.
5. Wiring ``compute_geometric_factors(..., angular_product=True)``, which is
   already written and **currently unreachable**.

Lo non-sunward species
^^^^^^^^^^^^^^^^^^^^^^

APID 1157 (``COD_LO_NSW_SPECIES_COUNTS``) has no ``process_l1a`` branch and no
species-name constant. The document's 8 NSW species (H+, He++, O5-8, C4-6,
Ne+Mg+Si, Fe, He+, CNO+) are not represented anywhere in the code.

At L2 the NSW isotropy factor (19 positions, averaged :math:`G_m` and
:math:`\varepsilon`) is *supported* by ``calculate_intensity`` via
``NSW_POSITIONS`` and ``average_across_positions=True`` - that constant is
defined but never passed.

Hi Appendix B: true omni-directional intensity
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The document is explicit that ``hi-omni`` as implemented is a **pseudo**
omni-directional intensity, and derives the solid-angle-corrected version from
the *sectored* intensities in Appendix B:

.. math::

   I_{\Omega}(i) = \frac{\sum_k \Omega_{12,k} \sum_n I(i, n, k)}{\Omega},
   \qquad \Omega = 1.956\ \mathrm{sr}

using per-pixel spin-integrated solid angles :math:`\Omega_k` (0.412282 to
1.17554 sr) and per-sector values :math:`\Omega_{12,k}` (0.05994 to 0.12354 sr).
None of these constants are in ``constants.py`` and the calculation does not
exist. Whether the SDC should produce it is a question for the CoDICE team.

Ratio uncertainties in I-ALiRT
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Section 14.2.2 defines pseudo-density uncertainties ``psU`` and quadrature
propagation into each ratio. ``calculate_ratios`` computes only the six ratios.

Frame conversion to spacecraft coordinates
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:math:`\theta_{SC} = (\theta_{inst} + 316^\circ) \bmod 360^\circ` is not applied
anywhere. All L2 angles are instrument-frame. This is consistent with the
document, which only needs the SC frame for L3 pitch angles, but it means an L2
consumer must do the rotation themselves.

Bugs
----

Ordered by how likely they are to affect released data.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Location
     - Issue
   * - ``codice_l2.py``, ``process_codice_l2``
     - **``UnboundLocalError`` for six descriptors.** The branch

       .. code-block:: python

          if dataset_name in ["imap_codice_l2_hi-counters-singles", ...]:
              pass

       matches, does nothing, and then falls through to
       ``if var in l2_dataset.data_vars`` - but ``l2_dataset`` was never
       assigned. Any attempt to run L2 for ``hi-counters-singles``,
       ``hi-counters-aggregated``, ``lo-counters-singles``,
       ``lo-counters-aggregated``, ``lo-sw-priority`` or ``lo-nsw-priority``
       crashes. Those six also have **no ``Logical_source``** entry, so even a
       fixed pass-through could not be written. The ``TODO`` above the branch
       says "May not even need L2 files for these products" - decide, then
       either delete the branch or finish it.
   * - ``codice_l2.py``, ``process_codice_l2``
     - Same ``UnboundLocalError`` for ``lo-nsw-species``, ``lo-sw-angular`` and
       ``lo-nsw-angular``: they match no branch at all. These *do* have
       ``Logical_source`` entries, so an operator can legitimately request them
       and get an unhelpful crash.
   * - ``codice_l1b.py``, ``convert_to_rates``
     - Same class of problem one level up. ``lo-sw-angular``,
       ``lo-nsw-angular`` and ``lo-nsw-species`` reach the ``lo-`` branch that
       computes ``energy_per_charge`` but match none of the three
       denominator branches, leaving ``denominator`` unbound.
   * - ``codice_l2.py``, ``process_hi_omni``
     - **Possible factor of 12.** The docstring says the denominator includes
       ``number_of_ssd``; the code does not multiply by it. The document's
       formula divides by :math:`\sum_k G_k = 12 \times 0.013 = 0.156`. Whether
       this is correct depends entirely on what the ``GF`` row of
       ``imap_codice_l2-hi-omni-efficiency_*.csv`` contains. **Verify before
       trusting absolute omni intensities.**
   * - ``codice_l2.py``, ``process_lo_direct_events``
     - **Elevation angle is looked up from ``apd_id``, not ``position``.**
       Document section 12.2.1 says "Converted from position to elevation
       angle". The same function uses ``position`` for the 13-24 spin shift, so
       the two fields are being used inconsistently within one function. For
       CoDICE-Lo the delay-line ``position`` is the direction measurement and
       ``apd_id`` identifies the energy detector - the document (section 4.1) is
       clear these are different things.
   * - ``utils.py``, ``get_codice_epoch_time``
     - Sub-seconds are divided by ``65536`` (2^16) for every product, but non-PHA
       science packets declare a **20-bit** ``Acq_Start_Subseconds`` field
       (PHA packets declare 16). If the field really is 20-bit, epochs are off
       by up to 15/16 of a second. The CoDICE team specified ``/ 65536``, so
       this may be intentional - but it should be written down.
   * - ``utils.py``, ``process_by_table_id``
     - ``view_id``, ``plan_id`` and ``plan_step`` are read from **record 0 only**
       and applied to the entire stream. Only ``table_id`` is grouped. A plan or
       view change mid-day silently unpacks the rest of the day with the wrong
       collapse table.
   * - ``codice_l1a_hi_omni.py``
     - Unexplained ``* 2`` in the sub-epoch spacing, flagged in the source as
       ``# TODO: why multiply by 2?``. It appears to undo the ``// 2`` in
       ``get_codice_epoch_time`` so that sub-epochs are spaced a full
       accumulation window apart, which is probably right - but it is
       load-bearing arithmetic with no justification.

Deviations from the algorithm document
--------------------------------------

These are design decisions, not bugs, but they will surprise anyone reading the
document first.

The 90 degree spin-angle correction
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Every spin-angle reference table in the code is the printed Rev 3 value minus
90 degrees.** This is a post-Rev-3 correction from Michael Starkey (issue #3242)
re-deriving instrument-frame reference angles relative to the instrument +X
axis. It affects ``L2_HI_SECTORED_ANGLE``, ``SSD_ID_TO_SPIN_ANGLE``,
``HI_IALIRT_REF_SPIN_ANGLE`` and the Lo direct-event ``+277.5`` constant.
``imap_processing/tests/codice/test_codice_spin_angles.py`` pins all of them as
exact-value regression tests specifically to stop somebody reverting to the PDF.
See :ref:`codice-spin-angle-offset`.

Epoch is the window centre, not the start
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Section 7 says "an Epoch variable is written ... which is the start time of the
acquisition ... a DELTA_EPOCH_PLUS is written". The code writes the **centre**
of the window with symmetric ``epoch_delta_minus`` / ``epoch_delta_plus`` in
integer nanoseconds. This is the IMAP project convention and applies to every
instrument.

Lo TOF conversion is quadratic
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Section 13.2.5 gives :math:`\tau_{ns} = 0.6217\,\tau_{ch} - 7.4437`. The code
uses a quadratic :math:`a\tau^2 + b\tau + c` with coefficients read from the
``l2-lo-onboard-mpq-cal`` ancillary file. The quadratic is the newer form; the
linear fit in the document is stale.

Negative TOF is filled at L2
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The document says nothing about this. The code sets ``tof_ns < 0`` to NaN in
``process_lo_direct_events``, with the comment that it mirrors "Menlo's L3a
handling" - i.e. the downstream L3 repository already discards them, and doing
it at L2 keeps the two consistent.

RGFO boundary fill applies to all dates
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Section 12.2.3 makes the "set ``half_spin == RGFO_half_spin`` to fill" rule
specific to P3 (2026-01-29 onwards). ``process_lo_species_intensity`` applies it
unconditionally. For P1/P2 data, where RGFO always triggered on half spin 0,
this NaNs the first half spin's worth of ESA steps regardless.

Hi geometric factor is read from the efficiency CSV
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The document quotes :math:`G_k = 0.013` cm2 sr as a fixed value from a SIMION
model. The code never hard-codes it; it reads a ``GF`` row from the efficiency
CSV. That is more flexible and more fragile - it means the geometric factor and
efficiency are versioned together and cannot be varied independently.

Aggregated counters variable lists are fixed
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Section 10.3.1 lists 30 selectable Lo rate types and notes "the CDF file which
contains this information must have all the variables defined, but only the
returned values will be filled". The code defines only the **nominal six**
(``LO_COUNTERS_AGGREGATED_VARIABLE_NAMES``). Which of those six are present is
read from the SCI-LUT, so turning one off is handled - but selecting a
*different* rate type in flight would require a code change. The same applies to
the seven Hi aggregated counters.

Direct events skip L1B
^^^^^^^^^^^^^^^^^^^^^^

There is no ``imap_codice_l1b_*-direct-events`` product. L2 direct-event
processing loads the **L1A** CDF. This matches the document.

FSW-version handling
--------------------

Two XTCE files exist for the 2026-01-29 FSW change, selected by parsing the date
from the L0 filename. Within a product, the branch is on ``packet_version``.
Both mechanisms are **whole-file**, and the known limitation is called out three
times in the source:

* ``codice_l1a.py:55`` - ``TODO get the exact time the FSW changed on january 29
  and relabel the xml file``. The switch is at midnight UTC, not the real
  changeover time.
* ``codice_l1a_lo_counters_singles.py:150`` and ``codice_l1a_lo_priority.py:186``
  - ``TODO handle boundary days where the FSW changed halfway through the
  dataset. E.g. Some packet_version = 1 and some = 2``. The code takes
  ``packet_versions[0]``.
* ``codice_l2.py:390`` - ``TODO: Fix this calculation on days when the sci Lut
  changes. There may be different packet versions in the same dataset.``

**2026-01-29 itself is therefore expected to be wrong** in some products, and so
is any future day on which the SCI-LUT changes mid-day. If you are chasing an
anomaly, check the date first.

Complete TODO inventory
-----------------------

.. list-table::
   :header-rows: 1
   :widths: 42 58

   * - Location
     - Text
   * - ``codice_l1a.py:55``
     - Get the exact FSW change time on 2026-01-29 and relabel the XML file.
   * - ``codice_l1a_de.py:455``
     - ``is this possible?`` - guard around an incomplete-event-packet case.
   * - ``codice_l1a_hi_omni.py:112``
     - ``why multiply by 2?`` in the sub-epoch spacing.
   * - ``codice_l1a_lo_counters_singles.py:150``
     - Handle boundary days with mixed ``packet_version``.
   * - ``codice_l1a_lo_priority.py:186``
     - Same.
   * - ``codice_l1b.py:113``
     - ``undo this when I get new validation file from Joey`` -
       ``acquisition_time_per_esa_step`` is temporarily kept in L1B output.
   * - ``codice_l2.py:390``
     - Geometric factors break on days when the SCI-LUT changes.
   * - ``codice_l2.py:494``
     - Pickup-ion geometric factor uses only position 0; the team wants this
       standardised.
   * - ``codice_l2.py:728``
     - Hi-omni L2 attribute workaround, "may go away once Joey and I fix L1B
       CDF".
   * - ``codice_l2.py:735``
     - L1B needs ``epoch_delta_plus`` / ``epoch_delta_minus`` attributes and an
       ``epoch`` dimension.
   * - ``codice_l2.py:1465``
     - Update the list of datasets that need geometric factors.
   * - ``codice_l2.py:1515``
     - "May not even need L2 files for these products" (the broken pass-through).
   * - ``constants.py:806``
     - Read the angular/priority/species variable name lists from the SCI-LUT
       instead of hard-coding them.

Test coverage
-------------

``imap_processing/tests/codice/``:

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - File
     - Coverage
   * - ``test_codice_l1a.py``
     - Housekeeping, Lo counters/priority/species, Hi counters/omni/sectored/
       priority, Lo and Hi direct events, incomplete segments. Compares against
       validation CDFs.
   * - ``test_codice_l1a_lut.py``
     - SCI-LUT JSON parsing: view table lookup, collapse pattern shape
       derivation.
   * - ``test_codice_l1b.py``
     - Rate conversion for Lo and Hi, uncertainty propagation.
   * - ``test_codice_l2.py``
     - Geometric factor and efficiency LUT reading, MPQ/TOF/energy conversions,
       Lo species intensity, Lo and Hi direct events.
   * - ``test_codice_hi_l2.py``
     - Hi omni and sectored intensities, spin-angle construction.
   * - ``test_codice_spin_angles.py``
     - Exact-value regression on every corrected reference angle. **Runs without
       external data** - the primary guard against reverting to the PDF tables.
   * - ``test_decompress.py``
     - All seven compression modes.
   * - ``test_process_by_table_id.py``
     - Table-ID grouping.

Validation inputs live under ``imap_processing/tests/codice/data/``
(``l0_data/``, ``l1a_input/``, ``l1b_validation/``, ``l1a_lut/``, ``l2_lut/``)
and are pinned in ``conftest.py`` by ``VALIDATION_FILE_DATE = "20250814"`` and
``VALIDATION_FILE_VERSION = "v015"``. ``conftest.codice_lut_path`` is a
``side_effect`` callable that stands in for
``ProcessingInputCollection.get_file_paths`` - **it raises ``ValueError`` on an
unknown descriptor**, which is the fastest way to discover which ancillary files
a new code path needs.

Untested paths worth knowing about:

* The 2026-01-29 packet definition is exercised only by a single
  ``fsw-changes`` fixture (``imap_codice_l0_raw_20260130_v001.pkts``).
* ``compute_geometric_factors(angular_product=True)`` is unreachable from
  production code and therefore untested against real data.
* The I-ALiRT CoDICE path is tested in ``imap_processing/tests/ialirt/``, not
  here.
