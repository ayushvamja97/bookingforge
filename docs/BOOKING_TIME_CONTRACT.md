# Booking-time prediction contract (research draft)

## Target and decision point
Predict whether a hotel booking is ultimately canceled or recorded as a no-show, using information plausibly known when the reservation is created. This project makes no operational policy recommendation to a real hotel.

## Dataset limitations
- Arrival dates exist, but no immutable booking-creation timestamp is provided. Reconstruct a **proxy** as arrival date minus `lead_time`.
- Feature values are not historical snapshots. Even seemingly booking-time attributes may have been edited later. The conservative allowlist in `src/bookingforge/cohort.py` is a **provisional research assumption**, not proof of feature availability.
- The file is selected by arrival period (2015-07 through 2017-08). Cohorts defined by reconstructed creation dates will therefore be incomplete, particularly near the right edge; many bookings created near that edge may arrive outside the source file. This selection bias must be investigated before treating cohort metrics as representative.
- Use `reservation_status_date` only to audit outcome maturity, never as a model feature.
- `is_canceled` is the binary label, not a predictor. `reservation_status` and `reservation_status_date` are forbidden predictors. `assigned_room_type`, `booking_changes`, `days_in_waiting_list` and other mutable columns are excluded from the strict baseline.
- Other booking-time features may be considered later only with explicit justification, a documented assumption and an ablation.

## Proposed initial split (not approved for modeling yet)
Reconstructed creation date through 2016-08-31: train; 2016-09-01 through 2016-12-31: validation; 2017-01-01 through 2017-08-31: test.

For a hypothetical model fit on 2016-08-31, training labels can only come from rows whose final status dates are on or before that date. Validation labels become available afterward and are used only for retrospective scoring. Do not train on a label whose observation date is in the future relative to the fit.

**Caution:** These proposed cohorts are particularly susceptible to selection/truncation because the source is arrival-bounded, not creation-bounded. A more defensible final retrospective protocol may split by arrival and enforce label-maturity rules for each fit date, while explicitly limiting its deployment claim. We will compare options before training.

## Before baseline training
1. Run `python src/bookingforge/cohort.py --csv data/raw/hotels.csv --output reports/creation_cohorts.json`.
2. Review cohort distributions and final-status maturity results for surprises.
3. Document the bias/selection limitation; if material, revise the evaluation design rather than conceal it.
4. Freeze exact features, split boundaries, missing-value rules, and training cutoff in a versioned experiment configuration.
5. Fit all imputers, encoders and scalers to training data only.
6. Prefer simple baselines and calibration. Preserve the final test partition for one final evaluation.

## Academic integrity and reproducibility
Re-implement concepts from personal coursework in independent modules without redistributing instructor-provided graded notebooks unless permitted. Keep raw data and individual booking records out of Git. Label synthetic capacity and revenue experiments as simulations.
