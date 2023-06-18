
import sys

def align(hangul, hanja):
    spaces = [i for i, sp in enumerate(hanja) if sp == ' ']
    hangul = list(hangul)
    hangul = [c for c in hangul if c != ' ']
    for index in spaces:
        hangul.insert(index, ' ')
    return ''.join(hangul)

def preprocess_file(input_file, output_file):
    with open(input_file, 'r') as f:
        text = f.read()

    data = text.split('\n')
    data = [line.split('\t') for line in data]
    output = [(align(hangul, hanja), hanja) for hangul, hanja in data]
    output = ['%s\t%s' % (hangul, hanja) for hangul, hanja in output]
    output = '\n'.join(output)
    with open(output_file, 'w') as f:
        f.write(output)

def main():
    args = sys.argv[1:]
    preprocess_file(args[0], args[1])

if __name__ == '__main__':
    main()
