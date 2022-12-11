evaluate_models:
	python src/evaluate_models.py --time-performance-iterations 100 --data-source SPY &&\
	python src/evaluate_models.py --time-performance-iterations 100 --data-source NDAQ

evaluate_finperf:
	python src/evaluate_models.py --data-source SPY &&\
	python src/evaluate_models.py --data-source NDAQ

evaluate_spy:
	python src/evaluate_models.py --time-performance-iterations 100 --data-source SPY

evaluate_ndaq:
	python src/evaluate_models.py --time-performance-iterations 100 --data-source NDAQ

plot_comps:
	python src/evaluate_models.py --time-performance-iterations 100 --data-source SPY --save-figs &&\
	python src/evaluate_models.py --time-performance-iterations 100 --data-source NDAQ --save-figs