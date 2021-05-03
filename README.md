# virus-trace-and-track
VTT is an on open source system for monitoring COVID-19 spread &amp; site visitor interaction  via CCTV cameras

### Modules

1. Viola Jones for Face Detection & Storage to search for visitors in the future - Done

	The `Viola Jones` directory contains a notebook file where the model has been trained and tested in.
	It also containes `num_feat_200.pkl` Which is the Viola Jones trained model with hyperparameter of T = 200
	The training code is written in multiprocess way to reduce training time, (The existing model took 1.5hrs to train on a 6 core CPU)
	2 files are ommited from the repo, since both exceeds 3 GB (The applied features pickle file). They won't however affect the pipeline, since the pipeline already creates them.

