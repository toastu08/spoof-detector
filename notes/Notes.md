# E1.1

### Hypothesis for this experiment

**The linear stft model will score the best**

My reasoning is that the model has learned tiny inaudible artifacts like biological noise (throat sounds and stuff like that) instead of speech patterns.

| Front-end     | Split | Attack subset | n (bonafide/spoof) |    EER |    AUC | FPR@0.5 | Spoof recall@0.5 |
| ------------- | ----- | ------------- | -----------------: | -----: | -----: | ------: | ---------------: |
| `linear_stft` | dev   | A01-A06       | 24844 (2548/22296) | 0.0569 | 0.9861 |  0.2005 |           0.9884 |
| `linear_stft` | eval  | A07-A19       | 71237 (7355/63882) | 0.1398 | 0.9371 |  0.1694 |           0.8796 |
| `mel128`      | dev   | A01-A06       | 24844 (2548/22296) | 0.0568 | 0.9876 |  0.2237 |           0.9878 |
| `mel128`      | eval  | A07-A19       | 71237 (7355/63882) | 0.1141 | 0.9536 |  0.1961 |           0.9232 |
| `mel256`      | dev   | A01-A06       | 24844 (2548/22296) | 0.0436 | 0.9919 |  0.1279 |           0.9878 |
| `mel256`      | eval  | A07-A19       | 71237 (7355/63882) | 0.1131 | 0.9525 |  0.1153 |           0.8880 |
| `mel64`       | dev   | A01-A06       | 24844 (2548/22296) | 0.1005 | 0.9669 |  0.2951 |           0.9746 |
| `mel64`       | eval  | A07-A19       | 71237 (7355/63882) | 0.1301 | 0.9434 |  0.2466 |           0.9244 |
| `mfcc40`      | dev   | A01-A06       | 24844 (2548/22296) | 0.1449 | 0.9327 |  0.4682 |           0.9664 |
| `mfcc40`      | eval  | A07-A19       | 71237 (7355/63882) | 0.1856 | 0.8963 |  0.3931 |           0.9027 |
| `randproj128` | dev   | A01-A06       | 24844 (2548/22296) | 0.1971 | 0.8847 |  0.6378 |           0.9632 |
| `randproj128` | eval  | A07-A19       | 71237 (7355/63882) | 0.1704 | 0.9064 |  0.5546 |           0.9368 |