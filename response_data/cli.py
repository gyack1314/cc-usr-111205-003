"""读取资料、执行带时间隔离的保序校准并输出JSON（不覆盖源文件）。"""
import argparse
import json

from .io import load
from .domain import summarize
from .calibration import analyze


def _parser():
    parser = argparse.ArgumentParser(description='防汛预警概率的分组保序校准')
    commands = parser.add_subparsers(dest='command', required=True)
    inspect = commands.add_parser('inspect', help='显示原始资料摘要')
    inspect.add_argument('file')
    calibrate = commands.add_parser(
        'calibrate', help='带时间隔离的开发内选参与外期复核')
    calibrate.add_argument('file')
    calibrate.add_argument('--output', help='把报告写到该路径；缺省打印到标准输出')
    return parser


def _dump(value):
    return json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)


def main():
    args = _parser().parse_args()
    value, digest = load(args.file)
    if args.command == 'inspect':
        print(_dump({'source_sha256': digest, 'summary': summarize(value)}))
        return
    report = analyze(value)
    report['source_sha256'] = digest
    text = _dump(report)
    if args.output:
        from pathlib import Path
        Path(args.output).write_text(text + '\n', encoding='utf8')
    else:
        print(text)


if __name__ == '__main__':
    main()
