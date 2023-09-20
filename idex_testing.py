from imap_processing.idex.idex_packet_parser import PacketParser
from imap_processing.idex.l2_processing import L2Processor

l0_file = "imap_processing/idex/tests/imap_idex_l0_20230725_v01-00.pkts"
l1_data = PacketParser(l0_file)
l1_cdf = l1_data.write_l1_cdf(version="01")
l2_data = L2Processor(l1_cdf)
l2_data.write_l2_cdf()