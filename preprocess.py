
import sys

hanja_numbers = set('〇零一壹壱二貳弍三叁參参四肆五伍六陸七柒八捌九玖十拾百佰千仟万萬億兆')

def inrange(c, a, b):
    return c >= a and c <= b

def preprocess_token(token):
    if all([inrange(c, '0', '9') for c in token]):
        return '<SRC>'
    elif all([inrange(c, 'A', 'Z') or inrange(c, 'a', 'z') for c in token]):
        return '<SRC>'
    elif all([c in hanja_numbers for c in token]):
        return '<SRC>'
    return token

def preprocess_line(line):
    tokens = line.split(' ')
    result = [preprocess_token(token) for token in tokens]
    return result

def preprocess_file(input_file, output_file):
    with open(input_file, 'r') as f:
        text = f.read()

    data = text.split('\n')
    data = [line.split('\t') for line in data]
    data = [(preprocess_line(hangul), preprocess_line(hanja)) for hangul, hanja in data]

    output = [(' '.join(hangul), ' '.join(hanja)) for hangul, hanja in data]
    output = ['%s\t%s' % (hangul, hanja) for hangul, hanja in output]
    output = '\n'.join(output)
    with open(output_file, 'w') as f:
        f.write(output)

def main():
    args = sys.argv[1:]
    preprocess_file(args[0], args[1])

if __name__ == '__main__':
    main()
