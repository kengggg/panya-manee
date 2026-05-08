You already reviewed the Panya Manee dashboard V1 plan.
Now do a second-pass convergence review using these counterpoints from Samantha.

Counterpoints / clarifications:
1. PRD Section 17 explicitly locks publication to a batch run result, not an arbitrary single run.
2. PRD Section 17 explicitly locks BOTH public speed metrics:
   - questions_per_min
   - correct_per_min
3. The repeat summary data already includes per-run:
   - questions_per_min
   - correct_per_min
   - latency_s
   So public speed metrics should be aggregated across the whole published batch, not derived from one canonical run.
4. A canonical run is still needed, but only for row-level examples/raw outputs, not for rank metrics.
5. PRD Section 17 explicitly locks downloadable raw JSON bundles for transparency. So this cannot be cut from V1, but it can be implemented minimally.
6. Best Small Model is still desirable in V1, but the current rule is too broad. Redefining it to `fits_comfortably_16gb` would make it non-redundant on the current model set.
7. Subject filter on the main leaderboard is low value and can be cut from initial release if needed.

Your task:
- Reconcile your previous critique with these constraints.
- Produce the best-ground final recommendation for V1.
- Be decisive.

Return these exact sections:

1. FINAL CONSENSUS VERDICT
- one paragraph max

2. FINAL V1 GROUND TRUTH
- list the locked decisions we should actually build against
- include metric definitions where needed

3. FINAL TEST PLAN
- only the must-have tests for V1
- grouped into:
  - generator/data
  - ranking/badges
  - examples/transparency
  - UI acceptance
- keep each test concrete and checkable

4. FINAL IMPLEMENTATION PLAN
- ordered phases
- explicitly identify what is in V1 vs deferred after launch

5. NON-NEGOTIABLE RISKS TO CLOSE BEFORE BUILDING
- short bullet list only

Important:
- do not re-open already locked PRD decisions unless they are impossible
- optimize for the smallest credible V1 that still honors the PRD
- do not introduce extra architecture for future-proofing
