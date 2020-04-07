import numpy as np

ATTACK_CAT_TO_ID = {
    'Normal': 0,
    'Reconnaissance': 1,
    'Backdoor': 2,
    'DoS': 3,
    'Exploits': 4,
    'Analysis': 5,
    'Fuzzers': 6,
    'Worms': 7,
    'Shellcode': 8,
    'Generic': 9,
}


def proto_to_one_hot(x):
    proto_val = [
        'udp', 'arp', 'tcp', 'igmp', 'ospf', 'sctp', 'gre', 'ggp', 'ip', 'ipnip', 'st2', 'argus', 'chaos', 'egp',
        'emcon', 'nvp', 'pup', 'xnet', 'mux', 'dcn', 'hmp', 'prm', 'trunk-1', 'trunk-2', 'xns-idp', 'leaf-1', 'leaf-2',
        'irtp', 'rdp', 'netblt', 'mfe-nsp', 'merit-inp', '3pc', 'idpr', 'ddp', 'idpr-cmtp', 'tp++', 'ipv6', 'sdrp',
        'ipv6-frag', 'ipv6-route', 'idrp', 'mhrp', 'i-nlsp', 'rvd', 'mobile', 'narp', 'skip', 'tlsp', 'ipv6-no', 'any',
        'ipv6-opts', 'cftp', 'sat-expak', 'ippc', 'kryptolan', 'sat-mon', 'cpnx', 'wsn', 'pvp', 'br-sat-mon', 'sun-nd',
        'wb-mon', 'vmtp', 'ttp', 'vines', 'nsfnet-igp', 'dgp', 'eigrp', 'tcf', 'sprite-rpc', 'larp', 'mtp', 'ax.25',
        'ipip', 'aes-sp3-d', 'micp', 'encap', 'pri-enc', 'gmtp', 'ifmp', 'pnni', 'qnx', 'scps', 'cbt', 'bbn-rcc', 'igp',
        'bna', 'swipe', 'visa', 'ipcv', 'cphb', 'iso-tp4', 'wb-expak', 'sep', 'secure-vmtp', 'xtp', 'il', 'rsvp',
        'unas', 'fc', 'iso-ip', 'etherip', 'pim', 'aris', 'a/n', 'ipcomp', 'snp', 'compaq-peer', 'ipx-n-ip', 'pgm',
        'vrrp', 'l2tp', 'zero', 'ddx', 'iatp', 'stp', 'srp', 'uti', 'sm', 'smp', 'isis', 'ptp', 'fire', 'crtp', 'crudp',
        'sccopmce', 'iplt', 'pipe', 'sps', 'ib', 'icmp', 'rtp'
    ]

    one_hot = [0] * len(proto_val)
    one_hot[proto_val.index(x)] = 1

    return one_hot


def service_to_one_hot(x):
    service_val = [
        '-', 'http', 'ftp', 'ftp-data', 'smtp', 'pop3', 'dns', 'snmp', 'ssl', 'dhcp', 'irc', 'radius', 'ssh'
    ]

    one_hot = [0] * len(service_val)
    one_hot[service_val.index(x)] = 1

    return one_hot


def state_to_one_hot(x):
    state_val = ['INT', 'FIN', 'REQ', 'ACC', 'CON', 'RST', 'CLO', 'ECO', 'PAR', 'URN', 'no']

    one_hot = [0] * len(state_val)
    one_hot[state_val.index(x)] = 1

    return one_hot


def double_sided_log(x):
    return np.sign(x) * np.log(1 + np.abs(x))


def sigmoid(x):
    return np.divide(1, (1 + np.exp(np.negative(x))))


def preprocess_data(data):
    # extract label and drop id
    attack_cat = data['attack_cat'].apply(func=(lambda x: ATTACK_CAT_TO_ID.get(x))).to_numpy().reshape(-1, 1)
    label = data['label'].to_numpy().reshape(-1, 1)
    data = data.drop(['id', 'attack_cat', 'label'], axis=1)

    # categorical to one-hot-encoding
    # extract and transform categorical to one-hot
    proto = np.array(data['proto'].apply(lambda x: proto_to_one_hot(x)).to_list())
    service = np.array(data['service'].apply(lambda x: service_to_one_hot(x)).to_list())
    state = np.array(data['state'].apply(lambda x: state_to_one_hot(x)).to_list())
    categorical = np.concatenate([proto, service, state], axis=1)
    data = data.drop(['proto', 'service', 'state'], axis=1)

    # transform to np matrix
    data = data.to_numpy()

    # merge categorical freature back to data
    data = np.concatenate([data, categorical], axis=1)

    # double-sided log: to attenuate outliers
    data = double_sided_log(data)

    # sigmoid transform
    data = sigmoid(data)

    # merge label back to data
    data = np.concatenate([data, attack_cat, label], axis=1)

    return data
