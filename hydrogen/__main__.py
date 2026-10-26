import argparse
from pathlib import Path
from . import reconstruct_hydrogen
from .validation import validate_history, plot_validation


def main():
    parser = argparse.ArgumentParser(description='Reconstruct six-state hydrogen on a prescribed HYDRAD atmosphere')
    parser.add_argument('root', type=Path)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--mode', choices=['equilibrium', 'time_dependent'], default='equilibrium')
    parser.add_argument('--atomic-data-root', type=Path)
    parser.add_argument('--stop-index', type=int)
    parser.add_argument('--output-stride', type=int, default=1)
    parser.add_argument('--input-stride', type=int, default=1, help='Subsample input history for cadence sensitivity tests')
    parser.add_argument('--max-step', type=float, default=.25)
    parser.add_argument('--initial', choices=['equilibrium', 'saved'], default='equilibrium')
    parser.add_argument('--validate', type=Path, help='Write validation JSON against held-out Hstate')
    parser.add_argument('--plot', type=Path)
    args = parser.parse_args()
    h = reconstruct_hydrogen(args.root, mode=args.mode, atomic_data_root=args.atomic_data_root,
                             stop_index=args.stop_index, output_stride=args.output_stride,
                             max_step=args.max_step, initial=args.initial, cache=args.cache,
                             input_stride=args.input_stride,
                             progress=lambda i,n,t: print(f'{i}/{n} profiles; t={t:g} s', flush=True))
    if args.validate:
        report = validate_history(h, args.root, output=args.validate)
        print({k:v for k,v in report.items() if k not in ('profiles_detail','reconstruction')})
    if args.plot:
        ids = h.indices[[0, len(h.indices)//3, 2*len(h.indices)//3, -1]]
        if args.validate and report['worst_profile'] is not None:
            ids[2] = report['worst_profile']['index']
        plot_validation(h, args.root, args.plot, list(dict.fromkeys(ids)))


if __name__ == '__main__':
    main()
