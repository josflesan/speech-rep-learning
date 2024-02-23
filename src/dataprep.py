import re
import kaldiark
import numpy


def load_mean_var(mean_var_path):
    f = open(mean_var_path)
    line = f.readline()
    sum = numpy.array([float(e) for e in line[1:-2].split(', ')])
    line = f.readline()
    sum2 = numpy.array([float(e) for e in line[1:-2].split(', ')])
    nsamples = int(f.readline())

    feat_mean = sum / nsamples
    feat_var = sum2 / nsamples - feat_mean * feat_mean

    return feat_mean, feat_var


def parse_scp_line(line):
    m = re.match(r'(.+) (.+):(.+)', line)
    key = m.group(1)
    file = m.group(2)
    shift = int(m.group(3))

    return key, file, shift


def load_label_dict(path):
    result = {}
    f = open(path)
    
    for i, line in enumerate(f):
        result[line.strip()] = i + 1

    f.close()

    return result


def load_labels(file, shift):
    f = open(file)
    f.seek(shift)
    labels = f.readline().split()
    f.close()

    return labels


def load_feat(file, shift, feat_mean=None, feat_var=None):
    f = open(file, 'rb')
    f.seek(shift)
    feat = kaldiark.parse_feat_matrix(f)
    f.close()

    if feat_mean is not None and feat_var is not None:
        feat = (feat - feat_mean) / numpy.sqrt(feat_var)

    return feat

