from odps import ODPS
from odps.tunnel import TableTunnel

odps = ODPS('LTAI9DpuxobuOxHZ', 'AVrnebwIMmF9PiIKxS3HrzkaL4E1cL', 'gravity_quant',
            endpoint='http://service.cn.maxcompute.aliyun.com/api')

def download_by_tunnel(table_name, file_path, row_count, pt=None, sep=','):
    """
    通过dataframe的方式读取odps的表数据
    :param table_name:
    :param file_path:
    :return:
    """

    tunnel = TableTunnel(odps)
    if pt is not None:
        download_session = tunnel.create_download_session(table_name, partition_spec=pt)
    else:
        download_session = tunnel.create_download_session(table_name)
    with open(file_path, 'w') as f:
        with download_session.open_record_reader(0, download_session.count) as reader:
            for record in reader:
                line = ''
                for i in range(row_count):
                    if i > 0:
                        line = line + sep
                    line = line + str(record[i])
                line = line + '\n'
                f.writelines(line)


if __name__ == '__main__':
    # download_by_tunnel('tdl_quant_day_history', 'test.csv', row_count=12, pt="pt = '2018-01-06'")
    download_by_tunnel('quant_index_local_history', 'Index/test.csv', pt="pt = '2018-07-04'", row_count=10)
    # download_by_tunnel('quant_day_local_history', 'Stock/test.csv', pt="pt = '2018-06-01'", row_count=48)