# -*- coding: utf-8 -*-
import numpy as np

# =============基本参数==============
Pload = np.array([500]*8760)
Hload = np.array([500]*8760)
Cload = np.array([500]*8760)
Cep = np.array([0.6]*8760)
Ces = np.array([0.4]*8760)
Ppvmax = np.array([0, 0, 0, 0, 0, 0.4, 0.4, 0.4, 0.4, 0.8, 0.8, 0.8, 0.4, 0.4, 0.4, 0.4, 0.4, 0, 0, 0, 0, 0, 0, 0]*365)
Pwtmax = np.array([0.3, 0.3, 0.3, 0.3, 0.3, 0.6, 0.6, 0.6, 0.6, 0.6, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.3, 0.3, 0.3, 0.6, 0.6, 0.6, 0.6, 0.6]*365)

# 风、光、蓄电池、吸收式制冷机、燃气轮机、锅炉、蓄热槽
# ============寿命================
LT_WT = 20  # 燃气轮机寿命/年
LT_PV = 20  # 光伏寿命
LT_PGU = 20  # 燃气轮机寿命
LT_AC = 20  # 吸收式制冷机寿命
LT_EC = 20  # 电制冷机寿命
LT_Bat = 10  # 蓄电池寿命
LT_GB = 10  # 燃气锅炉寿命
LT_HSS = 20  # 蓄热槽寿命

# ========设备出力参数===========
Pnet = np.zeros(8760)  # 微电网净出力
Pwt = np.zeros(8760)  # 风光出力
Ppv = np.zeros(8760)  # 光伏出力
Ppgu = np.zeros(8760)  # 燃气轮机电出力
Qpgu = np.zeros(8760)  # 燃气轮机热输入功率
Hpgu = np.zeros(8760)  # 燃气轮机热输出功率
Hpgu_r = np.zeros(8760)  # 燃气轮机实际功率
Hhss = np.zeros(8760)  # 储热设备出力
Cec = np.zeros(8760)  # 电制冷机出力
Cac = np.zeros(8760)  # 吸收式制冷机出力
Hac = np.zeros(8760)  # 吸收式制冷剂耗热量
Hgb = np.zeros(8760)  # 燃气锅炉实际功率
Hhot1 = np.zeros(8760)  # 燃气锅炉产热1
Hhot2 = np.zeros(8760)
Hgb_h = np.zeros(8760)  # 燃气锅炉实际功率
Hhot = np.zeros(8760)  # 热网净出力
Pec = np.zeros(8760)  # 电制冷机耗电
Peb = np.zeros(8760)  # 电锅炉制热耗电
Heb = np.zeros(8760)  # 电锅炉出力即 制热量
Pbat = np.zeros(8760)  # 1

# =============其他参数=============
e_pgu = 0.3  # 燃气轮机发电效率
h_pgu = 0.45  # 燃气轮机余热分流比制热 Note
c_pgu = 0.55  # 燃气轮机余热分流比制冷 Note
l_pgu = 0.2  # 燃气轮机热损失率
cop_ac = 1.3  # 吸收式制冷机能效比
e_gb = 0.9  # 燃气锅炉制热效率
cop_ec = 2.5  # 电制冷机能效比
wh_pgu = 0.5  # 燃气轮机余热回收率
h_wh = 0.95  # 余热回收制热效率
e_eb = 0.95  # 电锅炉制热效率
# 热损失率 + 余热回收率（制冷 + 制热）+发电 = 1

def CostFunction1(x):
    # ==============决策变量======================
    Uwt = x[0]  # 风机容量 %% 决策变量1
    Upv = x[1]  # 光伏容量 %% 决策变量2
    Ubat = x[2]  # 电池储能容量 %% 决策变量3
    Uac = x[3]  # 吸收式制冷机容量 %% 决策变量4
    Upgu = x[4]  # 燃气轮机容量 %% 决策变量5
    Ugb = x[5]  # 燃气锅炉容量 %% 决策变量6
    Uhss = x[6]  # 蓄热槽容量 %% 决策变量7
    # Uec = zeros(1, 1); % 电制冷机
    Uec = 250  # 电制冷机容量 % Note: 由于以电制冷、电制热托底，不再作为决策变量进行优化
    # Ueb = zeros(1, 1); %电锅炉
    Ueb = 250  # 电加热锅炉

    # =================电储能参数==================
    c_bat = 0.25  # Note

    # ========================热储能参数=================
    c_hs = 0.25  # Note写成公式，方便遗传算法调用
    # ====================资本回收系数===============
    z = 0.05
    y_10 = 10
    y_20 = 20
    s = 20
    Frk_10 = (1 / (1 + z) ** 10) * (z * (1 + z) ** y_10) / (((1 + z) ** y_10) - 1)  # 10年期资本回收系数 % 公式
    Frk_20 = (1 / (1 + z) ** 20) * (z * (1 + z) ** y_20) / (((1 + z) ** y_20) - 1)  # 20年期资本回收系数
    Fcr = (z * (1 + z) ** s) / ((1 + z) ** s - 1)  # 20年期资本回收因素

    # =========================投资成本=============
    Cwt_unit = 2800  # 修改 风机单位投资成本
    Cpv_unit = 2400  # 修改 光伏单位投资成本
    Cpgu_unit = 3000  # 燃气轮机单位投资成本
    Cbat_unit = 2000  # 蓄电池单位投资成本 容量成本
    Cac_unit = 1200  # 吸收式制冷机单位投资成本
    Cec_unit = 1100  # 电制冷机单位投资成本
    Cgb_unit = 1150  # 燃气锅炉单位投资成本
    Ceb_unit = 1050  # 电锅炉单位投资成本
    Chss_unit = 1603  # 蓄热槽投资成本

    #========================成本计算==========================
    # 成本计算
    #======================运行成本定义=========================
    ComPwt = np.zeros(8760)  # 风机运行成本
    ComPpv = np.zeros(8760)  # 光伏运行成本
    ComQpgu = np.zeros(8760)  # 燃气轮机运行成本
    ComCac = np.zeros(8760)  # 吸收式制冷机运行成本
    ComCec = np.zeros(8760)  # 电制冷机运行成本
    ComHeb = np.zeros(8760)  # 电锅炉运行成本
    ComHgb = np.zeros(8760)  # 电制冷机运行成本
    ComPcha = np.zeros(8760)  # 蓄电池运行成本
    ComPdischa = np.zeros(8760)  # 蓄电池运行成本
    ComHhss_in = np.zeros(8760)  # 蓄热槽运行成本
    ComHhss_out = np.zeros(8760)  # 蓄热槽运行成本

    #=======================置换成本==========================
    Cre_bat = Frk_10 * Fcr * Ubat * Cbat_unit  # 蓄电池置换成本
    Cre_hss = Frk_10 * Fcr * Uhss * Chss_unit  # 蓄热槽置换成本

    TotalComPwt = sum(ComPwt)  # 风机运维总成本 %%%% 去掉for1: 24循环，直接用sum求和函数
    TotalComPpv = sum(ComPpv)  # 光伏运维总成本
    TotalComQpgu = sum(ComQpgu)  # 燃气轮机运维总成本
    TotalComCac = sum(ComCac)  # 吸收式电制冷机运维总成本
    TotalComCec = sum(ComCec)  # 电制冷机运维总成本
    TotalComHgb = sum(ComHgb)  # 燃气锅炉运维总成本
    TotalComHeb = sum(ComHeb)  # 燃气锅炉运维总成本
    TotalComPbat = sum(ComPcha) + sum(ComPdischa)  # 蓄电池运维总成本
    TotalComHhss = sum(ComHhss_out) + sum(ComHhss_in)  # 蓄热槽运维总成本

    #=======================目标函数计算===========================
    Cre = Cre_bat + Cre_hss
    Com = TotalComPwt + TotalComPpv + TotalComQpgu + TotalComCac + TotalComCec + TotalComHgb + TotalComPbat + TotalComHhss + TotalComHeb#是否考虑初始投资的2 %？

    #ACS = Cinv + Com + Cre + TotalCostGas + TotalCostPbuy#目标函数1 # 系统年成本
    ATC = Com + Cre#目标函数1  #年成本：廉价-

    return ATC

def CostFunction2(x):
    # ==============决策变量======================
    Uwt = x[0]  # 风机容量 %% 决策变量1
    Upv = x[1]  # 光伏容量 %% 决策变量2
    Ubat = x[2]  # 电池储能容量 %% 决策变量3
    Uac = x[3]  # 吸收式制冷机容量 %% 决策变量4
    Upgu = x[4]  # 燃气轮机容量 %% 决策变量5
    Ugb = x[5]  # 燃气锅炉容量 %% 决策变量6
    Uhss = x[6]  # 蓄热槽容量 %% 决策变量7
    # Uec = zeros(1, 1); % 电制冷机
    Uec = 250  # 电制冷机容量 % Note: 由于以电制冷、电制热托底，不再作为决策变量进行优化
    # Ueb = zeros(1, 1); %电锅炉
    Ueb = 250  # 电加热锅炉

    # =================电储能参数==================
    c_bat = 0.25  # Note
    # 写成公式，方便遗传算法调用
    Pchamax = c_bat * Ubat  # 最大充电功率，采取储能容量 * 0，25的办法确定
    Pdismax = c_bat * Ubat  # 最大放电功率
    Pcha = np.zeros(8760)  # 充电功率
    e_cha = 0.9  # 充电效率
    Pdischa = np.zeros(8760)  # 放电功率
    e_discha = 0.9  # 放电效率
    SOCmax = 0.8  # 储能最大存储状态
    SOCmin = 0.2  # 储能最小存储状态
    SOC = np.zeros(8761)  # 储能荷电状态

    # ========================热储能参数=================
    c_hs = 0.25  # Note写成公式，方便遗传算法调用
    Hhss_in_max = c_hs * Uhss  # 最大充电功率，采取储能容量 * 0.25的办法确定
    Hhss_out_max = c_hs * Uhss  # 最大放电功率
    Hhss_in = np.zeros(8760)  # 充电功率
    e_loss = 0  # 热损失率
    e_hss_in = 0.9  # 热输入效率
    e_hss_out = 0.9  # 热输出效率
    Hhss_out = np.zeros(8760)  # 放电功率
    Hsocmax = 0.8  # 储热最大存储状态
    Hsocmin = 0.2  # 储热最小存储状态
    Hsoc = np.zeros(8761)  # 储能荷电状态

    # ================天然气消耗===================
    Lng = 9.78  # kwh / m3天燃气热值 （）
    Cgas = 2.75  # 工业管道天然气价格 / 元 % 统一价格
    Vpgu = np.zeros(8760)  # 燃气轮机消耗燃气体积
    Vgb = np.zeros(8760)  # 燃气锅炉消耗燃气体积
    Cvpgu = np.zeros(8760)  # 燃气轮机消耗燃气成本
    Cvgb = np.zeros(8760)  # 燃气锅炉消耗燃气成本

    # =====================co2消耗================
    G_co2 = 0.213  # 天然气燃烧排放因子kg / kWh
    E_co2 = 0.83  # 电网购电排放因子kg / kWh
    Emco2 = np.zeros(8760)  # 碳排放量kg

    # =========================购电成本======================
    Pbuy = np.zeros(8760)  # 向大电网购电量
    CostPbuy = np.zeros(8760)  # 向大电网购电成本
    Psell = np.zeros(8760)  # 向大电网出售电量
    CostPsell = np.zeros(8760)  # 向大电网出售获利
    C_gas_CH = np.zeros(8760)  # 燃气锅炉制热供给吸收式制冷机
    C_gas_C = np.zeros(8760)  # 燃气锅炉制热供给吸收式制冷机的出力
    E_C = np.zeros(8760)  # 等效冷功率
    E_CC = np.zeros(8760)  # 等效冷量
    Ee_CC = np.zeros(8760)  # 等效冷量由电制冷提供的耗电量
    # ==========================仿真程序-满足约束=============================
    '''
以冷定热： 在一个综合能源系统中，使用吸收式制冷机提供制冷服务。这种制冷机通过热力循环来产生制冷效果，
            同时也会产生余热。这余热可以被回收利用，用于供热系统，如暖气或热水。
            因此，制冷机提供的制冷量决定了系统中可用的热量，即以冷定热。
以热定电： 在同一个系统中，使用燃气轮机发电。燃气轮机在发电的过程中会产生热量，这些热量可以被用于提供热水或供暖。
            因此，系统中对热量的需求会影响燃气轮机的运行，热量需求增加时，
            燃气轮机可以增加发电量以提供更多的热量，即以热定电。
这种以冷定热、以热定电的耦合关系可以实现能源的高效利用，提高系统的能源利用率，减少能源浪费。
    '''
    for t in range(8760):
        # ========冷子系统==========
        if Uac >= Cload[t]:  # 吸收式制冷机容量大于冷负荷
            Cac[t] = Cload[t]  # Assumed: 吸收式制冷的出力，需要进一步讨论确定
        elif Uac < Cload[t]:
            Cac[t] = Uac
            Cec[t] = Cload[t] - Uac  # 不够的话电制冷来提供的冷量

        # 冷负荷首先来自吸收式制冷，吸收式制冷的热首先来自燃气轮机的余热，若不足来自燃气锅炉余热，若再不足这部分等效冷量直接来自电制冷 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        Hac[t] = Cac[t] / cop_ac  # 吸收式制冷需要的热源输入热量
        if Hac[t] <= Upgu * wh_pgu * c_pgu:
            C_gas_C[t] = 0  # 燃气锅炉制热
            Qpgu[t] = Hac[t] / c_pgu / wh_pgu
            Ppgu[t] = Qpgu[t] * e_pgu  # 燃气轮机发电
            Hpgu[t] = Qpgu[t] * wh_pgu * h_pgu * h_wh  # 燃气轮机, 燃气轮机余热回收制热效率0.95
        elif Upgu * wh_pgu * c_pgu <= Hac[t]:
            Qpgu[t] = Upgu
            Ppgu[t] = Qpgu[t] * e_pgu  # 燃气轮机发电
            Hpgu[t] = Upgu * wh_pgu * h_pgu * h_wh  # 燃气轮机, 燃气轮机余热回收制热效率0.95
            C_gas_C[t] = Hac[t] - Upgu * wh_pgu * c_pgu  # 中间变量
            if C_gas_C[t] <= Ugb:
                C_gas_C[t] = Hac[t] - Upgu * wh_pgu * c_pgu
            else:
                E_CC[t] = (C_gas_C[t] - Ugb) * cop_ac  # 等效冷功率
                C_gas_C[t] = Ugb
                Cac[t] = Cac[t] - E_CC[t]
                Hac[t] = Cac[t] / cop_ac
                Ee_CC[t] = E_CC[t] / cop_ec  # 等效冷量的耗电量

        # ========热子系统==========
        Hsoc[1] = 0.4
        SOC[1] = 0.4
        e_loss = 0.01  # 热高一点
        Hq = np.zeros(8760)
        # 中间变量-热净出力，先判断燃气轮机制热能否满足
        # ==========================储热================================
        if Hload[t] <= Hpgu[t]:  # revsied 直接用Hpgu(t)与Hload(t)对比，所需热量小于燃气轮机运行提供热量
            Hhot1[t] = Hpgu[t] - Hload[t]
            if Hhot1[t] <= Hhss_in_max:
                if (0.7855 * Hsocmin <= Hsoc[t]) and (Hsoc[t] <= Hsocmax):
                    Hhss_in[t] = min([Hhot1[t], (Hsocmax - Hsoc[t]) * Uhss / e_hss_in])  # 充放电功率从外部看
                    Hsoc[t + 1] = Hsoc[t] * (1 - e_loss) + Hhss_in[t] * e_hss_in / Uhss  # SOC是从内部看
                    Hq[t] = Hhot1[t] - Hhss_in[t]  # 弃热
                else:
                    Hhss_in[t] = 0  # 热储能充电功率
                    Hsoc[t + 1] = Hsoc[t] * (1 - e_loss) + Hhss_in[t] * e_hss_in / Uhss
                    Hq[t] = Hhot1[t] - Hhss_in[t]  # 多余的热弃掉

            elif Hhss_in_max <= Hhot1[t]:
                if (0.7855 * Hsocmin <= Hsoc[t]) and (Hsoc[t] <= Hsocmax):
                    Hhss_in[t] = min([Hhss_in_max, (Hsocmax - Hsoc[t]) * Uhss / e_hss_in])  # 热储能充电功率
                    Hsoc[t + 1] = Hsoc[t] * (1 - e_loss) + Hhss_in[t] * e_hss_in / Uhss
                    Hq[t] = Hhot1[t] - Hhss_in[t]
                else:
                    Hhss_in[t] = 0  # 热储能充电功率
                    Hsoc[t + 1] = Hsoc[t] * (1 - e_loss) + Hhss_in[t] * e_hss_in / Uhss
                    Hq[t] = Hhot1[t] - Hhss_in[t]  # 弃热
        # ==========================放热================================
        # 燃气轮机不够满足热需求
        if Hpgu[t] < Hload[t]:  # 燃气轮机不够满足热需求
            Hhot2[t] = Hload[t] - Hpgu[t]
            if Hhot2[t] <= Hhss_out_max:  # 首先考虑蓄热槽放热
                if (0.7855 * Hsocmin <= Hsoc[t]) and (Hsoc[t] <= Hsocmax):
                    Hhss_out[t] = min([Hhot2[t], (Hsoc[t] * Uhss - Hsocmin * Uhss) * e_hss_out])  # 放热
                    Hsoc[t + 1] = Hsoc[t] * (1 - e_loss) - (
                                Hhss_out[t] / e_hss_out) / Uhss  # %%%%%%%revised理解充放热功率与效率的关系
                    Hgb_h[t] = Hhot2[t] - Hhss_out[t]
                    Heb[t] = 0
                else:
                    Hhss_out[t] = 0
                    Hsoc[t + 1] = Hsoc[t] * (1 - e_loss) - (
                                Hhss_out[t] / e_hss_out) / Uhss  # %%%%%%%revised理解充放热功率与效率的关系
                    Hgb_h[t] = Hhot2[t] - Hhss_out[t]
            elif Hhss_out_max <= Hhot2[t]:
                if (0.7855 * Hsocmin <= Hsoc[t]) and (Hsoc[t] <= Hsocmax):
                    Hhss_out[t] = min([Hhss_out_max, (Hsoc[t] * Uhss - Hsocmin * Uhss) * e_hss_out])  # 放热
                    Hsoc[t + 1] = Hsoc[t] * (1 - e_loss) - (
                                Hhss_out[t] / e_hss_out) / Uhss  # %%%%%%%revised理解充放热功率与效率的关系
                    Hgb_h[t] = Hhot2[t] - Hhss_out[t]
                else:
                    Hhss_out[t] = 0
                    Hsoc[t + 1] = Hsoc[t] * (1 - e_loss) - (
                                Hhss_out[t] / e_hss_out) / Uhss  # %%%%%%%revised理解充放热功率与效率的关系 下同
                    Hgb_h[t] = Hhot2[t] - Hhss_out[t]
        # ==判断和燃气锅炉功率的关系,与Ugb判断,之前的讨论与上面冷子系统没有建立关系，是自己重新定义Hgbmnax来讨论？
        if (Hgb_h[t] + C_gas_C[t]) <= Ugb:
            Hgb[t] = Hgb_h[t] + C_gas_C[t]
            Heb[t] = 0
        else:
            Hgb[t] = Ugb
            Heb[t] = Hgb_h[t] + C_gas_C[t] - Ugb  # 多余的热由电加热锅炉补充
        Vgb[t] = (Hgb[t] / e_gb) / Lng  # m3
        Vpgu[t] = Qpgu[t] / Lng

        #CostPsell[t] = Psell[t] * Ces[t]
        #CostPbuy[t] = Pbuy[t] * Cep[t]
        #Pbat[t] = Pdischa[t] + Pcha[t]
    G_co2 = 0.213  # 天然气燃烧排放因子kg / kWh
    Emco2 = np.zeros(8760)  # co2总排放量
    Empgu = np.zeros(8760)  # 燃机co2排放量
    Emgb = np.zeros(8760)  # 锅炉co2排放量
    #EmGrid = np.zeros(8760)  # 电网购电co2排放量
    #==================相关成本计算=================
    for t in range(8760):
        Empgu[t] = G_co2 * Qpgu[t]  # 修改？燃气轮机的碳排放量
        Emgb[t] = G_co2 * (Hgb[t] / e_gb)  # 修改？燃气锅炉的碳排放量
        #EmGrid[t] = E_co2 * Pbuy[t]  # 向电网购电的部分，产生的间接碳排放量
        #Emco2[t] = Empgu[t] + Emgb[t] + EmGrid[t]  # co2排放量
        Emco2[t] = Empgu[t] + Emgb[t] # co2排放量

    TotalEmco2 = sum(Emco2)  # 二氧化碳总排放量
    CEV = TotalEmco2#目标函数2  # 碳排放：第一昂贵
    return CEV

def CostFunction3(x):
    # ==============决策变量======================
    Uwt = x[0]  # 风机容量 %% 决策变量1
    Upv = x[1]  # 光伏容量 %% 决策变量2
    Ubat = x[2]  # 电池储能容量 %% 决策变量3
    Uac = x[3]  # 吸收式制冷机容量 %% 决策变量4
    Upgu = x[4]  # 燃气轮机容量 %% 决策变量5
    Ugb = x[5]  # 燃气锅炉容量 %% 决策变量6
    Uhss = x[6]  # 蓄热槽容量 %% 决策变量7
    # Uec = zeros(1, 1); % 电制冷机
    Uec = 250  # 电制冷机容量 % Note: 由于以电制冷、电制热托底，不再作为决策变量进行优化
    # Ueb = zeros(1, 1); %电锅炉
    Ueb = 250  # 电加热锅炉

    # =================电储能参数==================
    c_bat = 0.25  # Note
    # 写成公式，方便遗传算法调用
    Pchamax = c_bat * Ubat  # 最大充电功率，采取储能容量 * 0，25的办法确定
    Pdismax = c_bat * Ubat  # 最大放电功率
    Pcha = np.zeros(8760)  # 充电功率
    e_cha = 0.9  # 充电效率
    Pdischa = np.zeros(8760)  # 放电功率
    e_discha = 0.9  # 放电效率
    SOCmax = 0.8  # 储能最大存储状态
    SOCmin = 0.2  # 储能最小存储状态
    SOC = np.zeros(8761)  # 储能荷电状态

    # ========================热储能参数=================
    c_hs = 0.25  # Note写成公式，方便遗传算法调用
    Hhss_in_max = c_hs * Uhss  # 最大充电功率，采取储能容量 * 0.25的办法确定
    Hhss_out_max = c_hs * Uhss  # 最大放电功率
    Hhss_in = np.zeros(8760)  # 充电功率
    e_loss = 0  # 热损失率
    e_hss_in = 0.9  # 热输入效率
    e_hss_out = 0.9  # 热输出效率
    Hhss_out = np.zeros(8760)  # 放电功率
    Hsocmax = 0.8  # 储热最大存储状态
    Hsocmin = 0.2  # 储热最小存储状态
    Hsoc = np.zeros(8761)  # 储能荷电状态

    # ================天然气消耗===================
    Lng = 9.78  # kwh / m3天燃气热值 （）
    Vpgu = np.zeros(8760)  # 燃气轮机消耗燃气体积
    Vgb = np.zeros(8760)  # 燃气锅炉消耗燃气体积

    # =========================购电成本======================
    Pbuy = np.zeros(8760)  # 向大电网购电量
    CostPbuy = np.zeros(8760)  # 向大电网购电成本
    Psell = np.zeros(8760)  # 向大电网出售电量
    CostPsell = np.zeros(8760)  # 向大电网出售获利
    C_gas_CH = np.zeros(8760)  # 燃气锅炉制热供给吸收式制冷机
    C_gas_C = np.zeros(8760)  # 燃气锅炉制热供给吸收式制冷机的出力
    E_C = np.zeros(8760)  # 等效冷功率
    E_CC = np.zeros(8760)  # 等效冷量
    Ee_CC = np.zeros(8760)  # 等效冷量由电制冷提供的耗电量
    # ==========================仿真程序-满足约束=============================
    '''
以冷定热： 在一个综合能源系统中，使用吸收式制冷机提供制冷服务。这种制冷机通过热力循环来产生制冷效果，
            同时也会产生余热。这余热可以被回收利用，用于供热系统，如暖气或热水。
            因此，制冷机提供的制冷量决定了系统中可用的热量，即以冷定热。
以热定电： 在同一个系统中，使用燃气轮机发电。燃气轮机在发电的过程中会产生热量，这些热量可以被用于提供热水或供暖。
            因此，系统中对热量的需求会影响燃气轮机的运行，热量需求增加时，
            燃气轮机可以增加发电量以提供更多的热量，即以热定电。
这种以冷定热、以热定电的耦合关系可以实现能源的高效利用，提高系统的能源利用率，减少能源浪费。
    '''
    for t in range(8760):
        # ========冷子系统==========
        if Uac >= Cload[t]:  # 吸收式制冷机容量大于冷负荷
            Cac[t] = Cload[t]  # Assumed: 吸收式制冷的出力，需要进一步讨论确定
        elif Uac < Cload[t]:
            Cac[t] = Uac
            Cec[t] = Cload[t] - Uac  # 不够的话电制冷来提供的冷量

        # 冷负荷首先来自吸收式制冷，吸收式制冷的热首先来自燃气轮机的余热，若不足来自燃气锅炉余热，若再不足这部分等效冷量直接来自电制冷 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        Hac[t] = Cac[t] / cop_ac  # 吸收式制冷需要的热源输入热量
        if Hac[t] <= Upgu * wh_pgu * c_pgu:
            C_gas_C[t] = 0  # 燃气锅炉制热
            Qpgu[t] = Hac[t] / c_pgu / wh_pgu
            Ppgu[t] = Qpgu[t] * e_pgu  # 燃气轮机发电
            Hpgu[t] = Qpgu[t] * wh_pgu * h_pgu * h_wh  # 燃气轮机, 燃气轮机余热回收制热效率0.95
        elif Upgu * wh_pgu * c_pgu <= Hac[t]:
            Qpgu[t] = Upgu
            Ppgu[t] = Qpgu[t] * e_pgu  # 燃气轮机发电
            Hpgu[t] = Upgu * wh_pgu * h_pgu * h_wh  # 燃气轮机, 燃气轮机余热回收制热效率0.95
            C_gas_C[t] = Hac[t] - Upgu * wh_pgu * c_pgu  # 中间变量
            if C_gas_C[t] <= Ugb:
                C_gas_C[t] = Hac[t] - Upgu * wh_pgu * c_pgu
            else:
                E_CC[t] = (C_gas_C[t] - Ugb) * cop_ac  # 等效冷功率
                C_gas_C[t] = Ugb
                Cac[t] = Cac[t] - E_CC[t]
                Hac[t] = Cac[t] / cop_ac
                Ee_CC[t] = E_CC[t] / cop_ec  # 等效冷量的耗电量

        # ========热子系统==========
        Hsoc[1] = 0.4
        SOC[1] = 0.4
        e_loss = 0.01  # 热高一点
        Hq = np.zeros(8760)
        # 中间变量-热净出力，先判断燃气轮机制热能否满足
        # ==========================储热================================
        if Hload[t] <= Hpgu[t]:  # revsied 直接用Hpgu(t)与Hload(t)对比，所需热量小于燃气轮机运行提供热量
            Hhot1[t] = Hpgu[t] - Hload[t]
            if Hhot1[t] <= Hhss_in_max:
                if (0.7855 * Hsocmin <= Hsoc[t]) and (Hsoc[t] <= Hsocmax):
                    Hhss_in[t] = min([Hhot1[t], (Hsocmax - Hsoc[t]) * Uhss / e_hss_in])  # 充放电功率从外部看
                    Hsoc[t + 1] = Hsoc[t] * (1 - e_loss) + Hhss_in[t] * e_hss_in / Uhss  # SOC是从内部看
                    Hq[t] = Hhot1[t] - Hhss_in[t]  # 弃热
                else:
                    Hhss_in[t] = 0  # 热储能充电功率
                    Hsoc[t + 1] = Hsoc[t] * (1 - e_loss) + Hhss_in[t] * e_hss_in / Uhss
                    Hq[t] = Hhot1[t] - Hhss_in[t]  # 多余的热弃掉

            elif Hhss_in_max <= Hhot1[t]:
                if (0.7855 * Hsocmin <= Hsoc[t]) and (Hsoc[t] <= Hsocmax):
                    Hhss_in[t] = min([Hhss_in_max, (Hsocmax - Hsoc[t]) * Uhss / e_hss_in])  # 热储能充电功率
                    Hsoc[t + 1] = Hsoc[t] * (1 - e_loss) + Hhss_in[t] * e_hss_in / Uhss
                    Hq[t] = Hhot1[t] - Hhss_in[t]
                else:
                    Hhss_in[t] = 0  # 热储能充电功率
                    Hsoc[t + 1] = Hsoc[t] * (1 - e_loss) + Hhss_in[t] * e_hss_in / Uhss
                    Hq[t] = Hhot1[t] - Hhss_in[t]  # 弃热
        # ==========================放热================================
        # 燃气轮机不够满足热需求
        if Hpgu[t] < Hload[t]:  # 燃气轮机不够满足热需求
            Hhot2[t] = Hload[t] - Hpgu[t]
            if Hhot2[t] <= Hhss_out_max:  # 首先考虑蓄热槽放热
                if (0.7855 * Hsocmin <= Hsoc[t]) and (Hsoc[t] <= Hsocmax):
                    Hhss_out[t] = min([Hhot2[t], (Hsoc[t] * Uhss - Hsocmin * Uhss) * e_hss_out])  # 放热
                    Hsoc[t + 1] = Hsoc[t] * (1 - e_loss) - (
                                Hhss_out[t] / e_hss_out) / Uhss  # %%%%%%%revised理解充放热功率与效率的关系
                    Hgb_h[t] = Hhot2[t] - Hhss_out[t]
                    Heb[t] = 0
                else:
                    Hhss_out[t] = 0
                    Hsoc[t + 1] = Hsoc[t] * (1 - e_loss) - (
                                Hhss_out[t] / e_hss_out) / Uhss  # %%%%%%%revised理解充放热功率与效率的关系
                    Hgb_h[t] = Hhot2[t] - Hhss_out[t]
            elif Hhss_out_max <= Hhot2[t]:
                if (0.7855 * Hsocmin <= Hsoc[t]) and (Hsoc[t] <= Hsocmax):
                    Hhss_out[t] = min([Hhss_out_max, (Hsoc[t] * Uhss - Hsocmin * Uhss) * e_hss_out])  # 放热
                    Hsoc[t + 1] = Hsoc[t] * (1 - e_loss) - (
                                Hhss_out[t] / e_hss_out) / Uhss  # %%%%%%%revised理解充放热功率与效率的关系
                    Hgb_h[t] = Hhot2[t] - Hhss_out[t]
                else:
                    Hhss_out[t] = 0
                    Hsoc[t + 1] = Hsoc[t] * (1 - e_loss) - (
                                Hhss_out[t] / e_hss_out) / Uhss  # %%%%%%%revised理解充放热功率与效率的关系 下同
                    Hgb_h[t] = Hhot2[t] - Hhss_out[t]
        # ==判断和燃气锅炉功率的关系,与Ugb判断,之前的讨论与上面冷子系统没有建立关系，是自己重新定义Hgbmnax来讨论？
        if (Hgb_h[t] + C_gas_C[t]) <= Ugb:
            Hgb[t] = Hgb_h[t] + C_gas_C[t]
            Heb[t] = 0
        else:
            Hgb[t] = Ugb
            Heb[t] = Hgb_h[t] + C_gas_C[t] - Ugb  # 多余的热由电加热锅炉补充
        Vgb[t] = (Hgb[t] / e_gb) / Lng  # m3
        Vpgu[t] = Qpgu[t] / Lng

        # ========电子系统==========
        Pwt = Uwt * Pwtmax  # 风机预计最大出力
        Ppv = Upv * Ppvmax  # 光伏预计最大出力
        Pec[t] = Cec[t] / cop_ec + Ee_CC[t]  # 电制冷耗电 + 吸收式制冷供热不足转为电制冷 因为电制冷以电托底，
        if Pec[t] <= Uec / cop_ec:
            pass
        else:
            Uec = Pec[t] * cop_ec  # 为了满足可以人为扩大
        if Heb[t] <= Ueb:
            pass
        else:
            Ueb = Heb[t]  # 为了满足可以人为扩大
        Peb[t] = Heb[t] / e_eb  # 电锅炉耗电
        Pnet[t] = Pwt[t] + Ppv[t] - Pload[t] - Pec[t] + Ppgu[t] - Peb[
            t]  # -Pehac(t); %电网净负荷 + 考虑多余的热转电Hche(t) * e_che, 这里燃气轮机耗电
        b_loss = 0.0001  # 自放电率
        SOC[1] = 0.4
        # =========================充电==============================
        if 0 <= Pnet[t] and Pnet[t] <= Pchamax:
            if (0.9975 * SOCmin <= SOC[t]) and (SOC[t] <= SOCmax):
                Pcha[t] = min([Pnet[t], (SOCmax * Ubat - SOC[t] * Ubat) / e_cha])  # 储充电功率 + 考虑充电效率
                SOC[t + 1] = SOC[t] * (1 - b_loss) + Pcha[t] * e_cha / Ubat
                Psell[t] = Pnet[t] - Pcha[t]
            else:
                Pcha[t] = 0  # 储能充电功率
                SOC[t + 1] = SOC[t] * (1 - b_loss) + Pcha[t] * e_cha / Ubat
                Psell[t] = Pnet[t] - Pcha[t]
        elif Pnet[t] > Pchamax:
            if (0.9975 * SOCmin <= SOC[t]) and (SOC[t] <= SOCmax):
                Pcha[t] = min([Pchamax, (SOCmax * Ubat - SOC[t] * Ubat) / e_cha])  # 热储能充电功率
                SOC[t + 1] = SOC[t] * (1 - b_loss) + Pcha[t] * e_cha / Ubat
                Psell[t] = Pnet[t] - Pcha[t]
            else:
                Pcha[t] = 0  # 储能充电功率
                SOC[t + 1] = SOC[t] * (1 - b_loss) + Pcha[t] * e_cha / Ubat
                Psell[t] = Pnet[t] - Pcha[t]

        # =========================放电==============================
        if (-Pchamax <= Pnet[t]) and (Pnet[t] <= 0):
            if (0.9975 * SOCmin <= SOC[t]) and (SOC[t] <= SOCmax):
                Pdischa[t] = min(abs(Pnet[t]), (SOC[t] * Ubat - SOCmin * Ubat) * e_discha)  # 热储能放电功率 + 考虑放电效率
                Pbuy[t] = -Pnet[t] - Pdischa[t]  # 购电
                SOC[t + 1] = SOC[t] * (1 - b_loss) - (Pdischa[t] / e_discha) / Ubat  # revised理解充放电功率与效率的关系 下同
            else:
                Pdischa[t] = 0
                Pbuy[t] = -Pnet[t] - Pdischa[t]  # 大电网购电
                SOC[t + 1] = SOC[t] * (1 - b_loss) - (Pdischa[t] / e_discha) / Ubat  # revised理解充放电功率与效率的关系 下同
        elif Pnet[t] < -Pchamax:
            if (0.9975 * SOCmin <= SOC[t]) and (SOC[t] <= SOCmax):
                Pdischa[t] = min(Pchamax, (SOC[t] * Ubat - SOCmin * Ubat) * e_discha)
                SOC[t + 1] = SOC[t] * (1 - b_loss) - (Pdischa[t] / e_discha) / Ubat  # revised 理解充放电功率与效率的关系 下同
                Pbuy[t] = -Pnet[t] - Pdischa[t]
            else:
                Pdischa[t] = 0
                Pbuy[t] = -Pnet[t] - Pdischa[t]
                SOC[t + 1] = SOC[t] * (1 - b_loss) - (Pdischa[t] / e_discha) / Ubat  # revised 理解充放电功率与效率的关系 下同

        CostPsell[t] = Psell[t] * Ces[t]
        CostPbuy[t] = Pbuy[t] * Cep[t]
        Pbat[t] = Pdischa[t] + Pcha[t]
    TotalPbuy = sum(Pbuy)

    SEI = TotalPbuy#目标函数3  # 购电率：第一昂贵
    #print('执行方法：CostFunction3')
    return SEI
#================问题===================
#如何把问题拆解，并根据x聚类，划分范围，
#根据值继承已经完成：
#+==============主要问题=====================
#如何在代中挑选部分解，用cost3来计算
#如何将剩余的解用伪值赋予
#如何更新我的种



