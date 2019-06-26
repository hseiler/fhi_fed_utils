from ..fhi_fed_utils import read_cfg


def test_config():
    path_cfg=''
    dict_path, dict_numerics = read_cfg(path_cfg)
    some_key = 'a key'
    assert isinstance(dict_path, dict)
    assert some_key in dict_numerics
