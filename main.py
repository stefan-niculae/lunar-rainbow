from multiprocessing import cpu_count, Pool

from functools import partial
from simulation import run_trial, parser


if __name__ == '__main__':
    args = parser.parse_args()

    if args.n_jobs == -1:
        args.n_jobs = cpu_count()

    if args.n_jobs > 1:
        p = Pool(args.n_jobs)
        p.map(partial(run_trial, args=args), range(args.n_jobs))
    else:
        run_trial(0, args)
