# Calibrate on your own data

The checkpoint's temperatures were fitted on its validation sets. If you act on a probability threshold (auto-approve
above 0.9, send to a human below), fit them again on a few hundred of your own labelled questions first. Why this
matters, and what it can and cannot change, is in [Calibration](../concepts/calibration.md).

## Fit a calibration

Give `calibrate` rows of a state, its questions and the right answers:

```python
cal = agent.calibrate(
    [
        {"state": {"image": img, "note": note}, "image_id": "a17",
         "questions": {"damage": damage_q, "outdoors": outdoors_q},
         "labels": {"damage": 2, "outdoors": False}},            # level index for score, option name for choice, bool for noul
        ...
    ],
    group_key="image_id",                                        # questions about one image stay together in every split
)
print(cal.summary())                                             # fitted temperatures; ECE raw / checkpoint / fitted, with 95% intervals
cal.save("my-calibration.json")
```

`calibrate` runs the model once over your rows and fits one temperature per question type. A type with fewer than 30
labelled questions shares one temperature fitted on all rows; with fewer than 30 in total the checkpoint's are kept.
Rows without the `group_key` field count as their own group, with a warning.

## Check that it helps

The fitted temperature is scored on rows it was not fitted on (5 folds that never split a group), next to the raw and
checkpoint temperatures, each with a 95% interval. `cal.evidence["all"]["ece_improvement"]` gives the paired interval
of the gain: if it includes 0, the new temperature is not shown to beat the checkpoint's on your data, and you can
keep the checkpoint's.

## Use it

```python
result = agent.predict(state, questions, calibration=laya.Calibration.load("my-calibration.json"))
result = agent.predict(state, questions, temperature={"noul": 1.4})   # or set one by hand, for this call only
```

Neither `temperature=` nor `calibration=` changes the agent: they apply to that one call.

- A calibration records which checkpoint it was fitted for, and warns if used with another
  (`strict_calibration=True` raises instead).
- Use the same `n_permutations` in `calibrate` and `predict`; the calibration records the value it was fitted with.

The fitting code is plain numpy in
[`laya/calibration.py`](https://github.com/r33drichards/laya-vision/blob/main/laya/calibration.py).
