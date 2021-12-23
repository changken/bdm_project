import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

class NUSW_NB15_Dataset(Dataset):
    def __init__(self, path):
        ds_type = path.split('/')[-1].split('-')[0]
        df = pd.read_csv(path)
        
#         x = df.drop(['id', 'attack_cat', 'label'], axis=1)
        x = df.drop(['id', 'attack_cat', 'label', 
                     'proto_3pc',
                     'proto_a/n',
                     'proto_aes-sp3-d',
                     'proto_argus',
                     'proto_aris',
                     'proto_ax.25',
                     'proto_bbn-rcc',
                     'proto_bna',
                     'proto_br-sat-mon',
                     'proto_cbt',
                     'proto_cftp',
                     'proto_chaos',
                     'proto_compaq-peer',
                     'proto_cphb',
                     'proto_cpnx',
                     'proto_crtp',
                     'proto_crudp',
                     'proto_dcn',
                     'proto_ddp',
                     'proto_ddx',
                     'proto_dgp',
                     'proto_egp',
                     'proto_eigrp',
                     'proto_emcon',
                     'proto_encap',
                     'proto_etherip',
                     'proto_fc',
                     'proto_fire',
                     'proto_ggp',
                     'proto_gmtp',
                     'proto_gre',
                     'proto_hmp',
                     'proto_i-nlsp',
                     'proto_iatp',
                     'proto_ib',
                     'proto_icmp',
                     'proto_idpr',
                     'proto_idpr-cmtp',
                     'proto_idrp',
                     'proto_ifmp',
                     'proto_igmp',
                     'proto_igp',
                     'proto_il',
                     'proto_ip',
                     'proto_ipcomp',
                     'proto_ipcv',
                     'proto_ipip',
                     'proto_iplt',
                     'proto_ipnip',
                     'proto_ippc',
                     'proto_ipv6',
                     'proto_ipv6-frag',
                     'proto_ipv6-no',
                     'proto_ipv6-opts',
                     'proto_ipv6-route',
                     'proto_ipx-n-ip',
                     'proto_irtp',
                     'proto_isis',
                     'proto_iso-ip',
                     'proto_iso-tp4',
                     'proto_kryptolan',
                     'proto_l2tp',
                     'proto_larp',
                     'proto_leaf-1',
                     'proto_leaf-2',
                     'proto_merit-inp',
                     'proto_mfe-nsp',
                     'proto_mhrp',
                     'proto_micp',
                     'proto_mobile',
                     'proto_mtp',
                     'proto_mux',
                     'proto_narp',
                     'proto_netblt',
                     'proto_nsfnet-igp',
                     'proto_nvp',
                     'proto_pgm',
                     'proto_pim',
                     'proto_pipe',
                     'proto_pnni',
                     'proto_pri-enc',
                     'proto_prm',
                     'proto_ptp',
                     'proto_pup',
                     'proto_pvp',
                     'proto_qnx',
                     'proto_rdp',
                     'proto_rsvp',
                     'proto_rtp',
                     'proto_rvd',
                     'proto_sat-expak',
                     'proto_sat-mon',
                     'proto_sccopmce',
                     'proto_scps',
                     'proto_sdrp',
                     'proto_secure-vmtp',
                     'proto_sep',
                     'proto_skip',
                     'proto_sm',
                     'proto_smp',
                     'proto_snp',
                     'proto_sprite-rpc',
                     'proto_sps',
                     'proto_srp',
                     'proto_st2',
                     'proto_stp',
                     'proto_sun-nd',
                     'proto_swipe',
                     'proto_tcf',
                     'proto_tlsp',
                     'proto_tp++',
                     'proto_trunk-1',
                     'proto_trunk-2',
                     'proto_ttp',
                     'proto_uti',
                     'proto_vines',
                     'proto_visa',
                     'proto_vmtp',
                     'proto_vrrp',
                     'proto_wb-expak',
                     'proto_wb-mon',
                     'proto_wsn',
                     'proto_xnet',
                     'proto_xns-idp',
                     'proto_xtp',
                     'proto_zero',
                     'service_dhcp',
                     'service_irc',
                     'service_radius',
                     'service_ssl',
                     'state_ACC',
                     'state_CLO',
                     'state_ECO',
                     'state_PAR',
                     'state_RST',
                     'state_URN',
                     'state_no'], axis=1)
        y = pd.Categorical(df['attack_cat']).codes.astype(np.float64)

        self.x = torch.Tensor(x.to_numpy())
        self.y = torch.Tensor(y).to(dtype=torch.long)

        self.dim = self.x.shape[1]

        print(
            f'Finished reading the {ds_type} set of Dataset '\
            f'({len(self.x)} samples found, each dim = {self.dim})'
        )

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.x)
    

def prep_dataloader(path, batch_size, shuffle):
    dataset = NUSW_NB15_Dataset(path)
    dataloader = DataLoader(
        dataset,
        batch_size,
        shuffle
    )
    return dataloader