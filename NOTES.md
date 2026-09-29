# Notes

Context and notes from Jules and Caleb, including meeting notes.

## Meeting notes — 2026-09-29

### Caleb

- Increase the dataset size.
- Look at the model weights.
- Synthetic injection: add synthetic objects to the validation images, e.g. the ruins of a base or something similar.
- Give feedback on Mk18, the hyperparameters and the new datasets:
  - https://huggingface.co/datasets/jules1235813/xenarch-hirise
  - https://huggingface.co/datasets/jules1235813/xenarch-lroc-nac

### Second-stage classifier

Once the anomaly detection model has found the top anomalies, a second model could check them. It would be trained on labeled data or on artificial structures, and it would sort the candidates and test whether each one is a true anomaly.

## Ideas

- Skyfall: a possible partnership with Caltech for this model.
- NASA Ames: work on caves.
