# Dashboard Crash Test Checklist

## Visual/UI checks

- [ ] Dashboard loads at `http://localhost:8000/dashboard`
- [ ] Cards are aligned and readable on a laptop screen
- [ ] Prediction form is readable and well spaced
- [ ] Output is visual (cards/charts), not raw JSON blocks
- [ ] Evidence section is card-based with availability indicators

## MLflow checks

- [ ] Start stack and open MLflow UI `http://localhost:5000`
- [ ] Open dashboard and verify MLflow status badges/cards are visible
- [ ] Stop MLflow and refresh dashboard
- [ ] Verify dashboard does not crash and MLflow status becomes unavailable
- [ ] Verify missing model registry does not crash dashboard
- [ ] Verify missing run metrics/artifacts do not crash dashboard

## Prediction checks

- [ ] Run valid single prediction
- [ ] Run invalid single prediction (missing required field)
- [ ] Run wrong type prediction (`hr="bad"`)
- [ ] Run out-of-range prediction (`temp=1.5`)
- [ ] Repeatedly click prediction button (loading/disable behavior)
- [ ] Click **Reset**
- [ ] Click **Load Safe Demo Example**

## Visualization checks

- [ ] Confidence gauge updates after prediction
- [ ] Feature bars update for `temp`, `atemp`, `hum`, `windspeed`
- [ ] Hourly scenario chart appears
- [ ] Batch chart appears
- [ ] Recent predictions chart updates over runtime

## Crash checks

- [ ] Empty batch request rejected cleanly
- [ ] Batch size >100 rejected with clean message
- [ ] Malformed JSON request returns clean validation error
- [ ] Missing `drift_summary.json` does not crash dashboard/API
- [ ] Missing evidence file appears as unavailable, no crash
- [ ] Model unavailable case keeps dashboard alive and reports not ready
- [ ] Repeated refresh does not break layout or scripts
- [ ] `/health` responds
- [ ] `/ready` responds
- [ ] `/metrics` responds
