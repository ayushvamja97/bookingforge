# Dataset and prediction contract — pending validation

**Proposed task:** predict cancellation risk for a reservation at a precisely defined observation time.

## Unresolved before training
- Select observation time: reservation creation vs a fixed number of days before arrival.
- Verify how booking creation and arrival dates are reconstructed from dataset fields.
- Determine whether each candidate feature was actually available at observation time.
- Ensure all training labels would be observable at each chronological cutoff (label maturity).
- Determine whether arrivals are a valid chronological proxy for the intended task.
- Validate class distributions and missingness for each hotel and split.
- Verify data redistribution terms; store download instructions, not raw guest records.

## Known forbidden features
`is_canceled`, `reservation_status`, `reservation_status_date` are outcome fields and must not be predictors.

## Important warning
The current `validate_features` function only blocks known outcome columns. It is NOT a complete temporal leakage detector and must not be treated as approval of other fields.

## Proposed provisional split (not validated)
Train: arrival <= 2016-08-31; validation: 2016-09-01 to 2016-12-31; test: 2017-01-01 to 2017-08-31. This is provisional until the prediction-time and label-maturity audit is complete.
