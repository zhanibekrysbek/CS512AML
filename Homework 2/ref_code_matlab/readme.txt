ex_1c.m: shows how the random instance was generated, and how the reference answer decode_output.txt was derived. 
ex_2a.m: shows how the random instance was generated, and how the reference answer gradient.txt was derived.
ex_2b.m: shows how the reference answers solution.txt and prediction.txt were derived.

decode.m: implements the MAP inference.
crf_obj.m: implements the objective value and gradient of CRF.
crf_test.m: implements the performance evaluation on test data.

The entry point of benchmarking code is all_experiment.m.  The intermediate result of CRF training is stored in monitor.txt, based on which plot_curve.m plots the curves.
