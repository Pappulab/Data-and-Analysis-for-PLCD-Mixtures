'''
Created on Oct 23, 2020

@author: mina
'''
import sys
import os
import math
import numpy as np
import argparse
import traj_analyzer_multi_new
import concave_hull_ana
import msd_ana

def MKDirCatch(this_dir):
    try:
        os.mkdir(this_dir)
    except OSError as myErr:
        if myErr.errno != 17:
            print("There is something wrong!")
            raise

def Temp_Path(path, temp):
    return path + 'T' + '%.1f' %temp + '/'

def Box_Path(path, box):
    return path + 'B' + '%i' %box + '/'

def Pre_Box_Path(path, pre_box):
    return path + 'P' + '%i' %pre_box + '/'

def Chains_Path(path, chains):
    return path + 'C' + '%i' %chains + '/'

def Run_Path(path, run_it):
    return path + str(run_it + 1) + '/'

def Print_Func(text):
    with open('log_file.txt', 'a') as myfile:
        myfile.write(text + '\n\n')

if __name__=="__main__":
    
    parser = argparse.ArgumentParser()

    parser.add_argument("-prefix", help="current prefix")
    parser.add_argument("-name", help="current name")
    parser.add_argument("-binary", type = int, help="1 if trj file is in .lassi format. 0 if trj file is in .lammpstrj format")
    parser.add_argument("-ana_dir", help="analysis directory")
    parser.add_argument("-r_dir", help="repeat directory")
    parser.add_argument("-energy", help="current energy")
    parser.add_argument("-temp", type = float, nargs='+', help="temp list for structure")
    parser.add_argument("-box", type = int, nargs='+', help="box list")
    parser.add_argument("-pre_box", type = int, nargs='+', help="pre_box list")
    parser.add_argument("-dist_chains", type = int, help="number of distinct types of chains")
    parser.add_argument("-chains", type = int, nargs='+', help="chains list")
    parser.add_argument("-chains_index", type = int, nargs='+', help="chains index list; used if num_chains > 0")
    parser.add_argument("-runs", type = int, help="number of replicates")
    parser.add_argument("-time_params", type = int, nargs='+', help="list of inputs describing simulation timesteps (time_start, time_increment, time_end, time_cutoff).\
                                                              time_start is the first time in the trj file, time_inc is the delta between frames, time_end is the\
                                                              final time in the trj file, time_cutoff is the time above which we should perform the analysis.")
    parser.add_argument("-stickers", type = int, nargs='+', help="list of beads for various calculations")
    parser.add_argument("-fractal_ana_inputs", type = int, nargs='+', help="list of inputs for fractal analysis (#samples, box_radius, max_distance_from_COM)")
    parser.add_argument("-concave_hull_ana_inputs", type = float, nargs='+', help="list of inputs for fractal analysis (#thickness, requirement, alpha, theta)")
    parser.add_argument("-analyses", type = int, nargs='+', help="analysis list w/ order nu_check, re_dist_check, gtensor_dist_check, fractal_ana_dist_check,\
                                                                  com_dist_check, rdf_dist_check, mol_com_dist_check, sticker_cluster_dist_check,\
                                                                  crosslink_dist_check, solvation_dist_check, com_gtensor_dist_check, connectedness_dist_check,\
                                                                  sticker_lifetime_dist_check, concave_hull_dist_check, concave_hull_ana_check, mol_com_ana_check.\
                                                                  1 to perform analysis, 0 to ignore analysis.")
    
    args = parser.parse_args()
    
    prefix = args.prefix
    name = args.name
    binary_check = args.binary
    ana_path = args.ana_dir
    repeat_dir = args.r_dir
    energy = args.energy
    cur_temp_list = args.temp
    box_list = args.box
    pre_box_list = args.pre_box
    dist_chains = args.dist_chains
    chains_list = args.chains
    chains_index_list = args.chains_index
    runs_per_condition = args.runs
    time_start, time_inc, time_end, time_cutoff = args.time_params
    sticker_list = args.stickers
    fractal_ana_list = args.fractal_ana_inputs
    concave_hull_ana_list = args.concave_hull_ana_inputs
    nu_check, re_dist_check, gtensor_dist_check, fractal_ana_dist_check,\
    com_dist_check, rdf_dist_check, mol_com_dist_check, sticker_cluster_dist_check,\
    crosslink_dist_check, solvation_dist_check, com_gtensor_dist_check,\
    connectedness_dist_check, sticker_lifetime_dist_check,\
    concave_hull_dist_check, concave_hull_ana_check, mol_com_ana_check = args.analyses
    
    trj_check = not all(v == 0 for v in args.analyses[:-2])
    
    if dist_chains > 1:
        chains_list = [chains_list[i: i + dist_chains] for i in range(0, len(chains_list), dist_chains)]
    for temp in cur_temp_list:
        if energy == 'NONE' and temp != cur_temp_list[0]:
            continue
        temp_dir = Temp_Path(repeat_dir, temp)
        for box in box_list:
            box_dir = Box_Path(temp_dir, box)
            for pre_box in pre_box_list:
                pre_box_dir = Pre_Box_Path(box_dir, pre_box)
                for chain_index, chains in enumerate(chains_list):
                    if dist_chains > 1:
                        cur_chains_list = []
                        cur_dist_chains = 0
                        for cur_chains in chains:
                            if int(cur_chains) > 0:
                                cur_dist_chains += 1
                                cur_chains_list.append(int(cur_chains))
                        cur_chains_array = np.array(cur_chains_list, dtype='int')
                        cur_chain_index = chains_index_list[chain_index]
                    else:
                        cur_dist_chains = 1
                        cur_chains_array = np.array([int(chains)], dtype='int')
                        cur_chain_index = chains
                    num_chains = np.sum(cur_chains_array)
                    chains_dir = Chains_Path(pre_box_dir, cur_chain_index)

                    file_name_nu = ana_path + name + '_' + str(temp) + '_' + str(box) + '_' + str(pre_box) + '_' + str(cur_chain_index) + '_Nu.dat'
                    file_name_rg_full = ana_path + name + '_' + str(temp) + '_' + str(box) + '_' + str(pre_box) + '_' + str(cur_chain_index) + '_Rg.dat'
                    file_name_re_dist_clus = ana_path + name + '_' + str(temp) + '_' + str(box) + '_' + str(pre_box) + '_' + str(cur_chain_index) + '_Re_Distribution_Clus.dat'
                    file_name_re_dist_solo = ana_path + name + '_' + str(temp) + '_' + str(box) + '_' + str(pre_box) + '_' + str(cur_chain_index) + '_Re_Distribution_Solo.dat'
                    file_name_gtensor_dist_clus = ana_path + name + '_' + str(temp) + '_' + str(box) + '_' + str(pre_box) + '_' + str(cur_chain_index) + '_GTensor_Distribution_Clus.dat'
                    file_name_fractal_ana_dist_clus = ana_path + name + '_' + str(temp) + '_' + str(box) + '_' + str(pre_box) + '_' + str(cur_chain_index) + '_Fractal_Ana_Distribution_Clus.dat'
                    file_name_rg_COM = ana_path + name + '_' + str(temp) + '_' + str(box) + '_' + str(pre_box) + '_' + str(cur_chain_index) + '_Rg_COM_Distribution.dat'
                    file_name_COM_bin_count = ana_path + name + '_' + str(temp) + '_' + str(box) + '_' + str(pre_box) + '_' + str(cur_chain_index) + '_COM_bin_count_Distribution.dat'
                    file_name_COM_bead = ana_path + name + '_' + str(temp) + '_' + str(box) + '_' + str(pre_box) + '_' + str(cur_chain_index) + '_COM_bead_Distribution.dat'
                    file_name_COM_chain = ana_path + name + '_' + str(temp) + '_' + str(box) + '_' + str(pre_box) + '_' + str(cur_chain_index) + '_COM_chain_Distribution.dat'
                    file_name_COM_sticker = ana_path + name + '_' + str(temp) + '_' + str(box) + '_' + str(pre_box) + '_' + str(cur_chain_index) + '_COM_sticker_Distribution.dat'
                    file_name_COM_Re = ana_path + name + '_' + str(temp) + '_' + str(box) + '_' + str(pre_box) + '_' + str(cur_chain_index) + '_COM_Re_Distribution.dat'
                    file_name_COM_angle = ana_path + name + '_' + str(temp) + '_' + str(box) + '_' + str(pre_box) + '_' + str(cur_chain_index) + '_COM_cosine_Distribution.dat'
                    file_name_COM_dot_prod = ana_path + name + '_' + str(temp) + '_' + str(box) + '_' + str(pre_box) + '_' + str(cur_chain_index) + '_COM_dot_prod_Distribution.dat'
                    file_name_COM_solvation = ana_path + name + '_' + str(temp) + '_' + str(box) + '_' + str(pre_box) + '_' + str(cur_chain_index) + '_COM_solvation_Distribution.dat'
                    file_name_RDF = ana_path + name + '_' + str(temp) + '_' + str(box) + '_' + str(pre_box) + '_' + str(cur_chain_index) + '_RDF_Distribution.dat'
                    file_name_mol_COM = ana_path + name + '_' + str(temp) + '_' + str(box) + '_' + str(pre_box) + '_' + str(cur_chain_index) + '_mol_COM_Distribution.dat'
                    file_name_mol_COM_2 = ana_path + name + '_' + str(temp) + '_' + str(box) + '_' + str(pre_box) + '_' + str(cur_chain_index) + '_mol_COM_Distance_Distribution.dat'
                    file_name_mol_COM_3 = ana_path + name + '_' + str(temp) + '_' + str(box) + '_' + str(pre_box) + '_' + str(cur_chain_index) + '_mol_COM_Binned_Distance_Distribution.dat'
                    file_name_mol_COM_ana = ana_path + name + '_' + str(temp) + '_' + str(box) + '_' + str(pre_box) + '_' + str(cur_chain_index) + '_mol_COM_Average.dat'
                    file_name_displacement = ana_path + name + '_' + str(temp) + '_' + str(box) + '_' + str(pre_box) + '_' + str(cur_chain_index) + '_Displacement_Distribution.dat'
                    file_name_sticker_cluster = ana_path + name + '_' + str(temp) + '_' + str(box) + '_' + str(pre_box) + '_' + str(cur_chain_index) + '_sticker_cluster_Distribution.dat'
                    file_name_crosslink = ana_path + name + '_' + str(temp) + '_' + str(box) + '_' + str(pre_box) + '_' + str(cur_chain_index) + '_Crosslink.dat'
                    file_name_cluster = ana_path + name + '_' + str(temp) + '_' + str(box) + '_' + str(pre_box) + '_' + str(cur_chain_index) + '_cluster_Distribution.dat'
                    file_name_solvation = ana_path + name + '_' + str(temp) + '_' + str(box) + '_' + str(pre_box) + '_' + str(cur_chain_index) + '_solvation_Distribution.dat'
                    file_name_COM_gtensor = ana_path + name + '_' + str(temp) + '_' + str(box) + '_' + str(pre_box) + '_' + str(cur_chain_index) + '_COM_GTensor_Distribution.dat'
                    file_name_connectedness = ana_path + name + '_' + str(temp) + '_' + str(box) + '_' + str(pre_box) + '_' + str(cur_chain_index) + '_Connectedness_Distribution.dat'
                    file_name_connectedness_binned = ana_path + name + '_' + str(temp) + '_' + str(box) + '_' + str(pre_box) + '_' + str(cur_chain_index) + '_Connectedness_Binned_Skip_0_Distribution.dat'
                    file_name_connectedness_full = ana_path + name + '_' + str(temp) + '_' + str(box) + '_' + str(pre_box) + '_' + str(cur_chain_index) + '_Connectedness_Full_Skip_0_Distribution.dat'
                    file_name_sticker_lifetime = ana_path + name + '_' + str(temp) + '_' + str(box) + '_' + str(pre_box) + '_' + str(cur_chain_index) + '_Sticker_Lifetime_Distribution.dat'
                    file_name_concave_hull = ana_path + name + '_' + str(temp) + '_' + str(box) + '_' + str(pre_box) + '_' + str(cur_chain_index) + '_Concave_Hull_Distribution.dat'
                    file_name_concave_hull_fft = ana_path + name + '_' + str(temp) + '_' + str(box) + '_' + str(pre_box) + '_' + str(cur_chain_index) + '_Concave_Hull_FFT.dat'
                    
                    if trj_check == True:
                        
                        if nu_check == 1:
                            with open(file_name_nu, 'w+') as fp:
                                fp.write('#Chain_Type\tNu_Dense\tNu_Dense_Error\tNu_Dilute\tNu_Dilute_Error\n')
                            with open(file_name_rg_full, 'w+') as fp:
                                fp.write('#Chain_Type\tRg_Dense\tRg_Dilute\n')
                
                        if re_dist_check == 1:
                            with open(file_name_re_dist_clus, 'w+') as fp:
                                fp.write('#Chain_Type\tRe\n')
                            with open(file_name_re_dist_solo, 'w+') as fp:
                                fp.write('#Chain_Type\tRe\n')
            
                        if gtensor_dist_check == 1:
                            with open(file_name_gtensor_dist_clus, 'w+') as fp:
                                fp.write('#Cluster gyration tensor with order xx, yy, zz, xy, yz, zx\n')
            
                        if fractal_ana_dist_check == 1:
                            with open(file_name_fractal_ana_dist_clus, 'w+') as fp:
                                fp.write('#Parameters: samples/frame %i, cube side-length %i, max distance %i, \n' % (fractal_ana_list[0], fractal_ana_list[1] * 2 + 1, fractal_ana_list[2]))
                                fp.write('#X-coord\tY-coord\tZ-coord\tDistance from COM\t# empty sites\t# sticker sites\t# empty sticker neighbors\t# sticker sticker neighbors\t# spacer sticker neighbors\n')
                            
                        if com_dist_check == 1:
                            with open(file_name_rg_COM, 'w+') as fp:
                                fp.write('#COM_Bin\tRg_Squared\n')
                                fp.write('#Order is chain type\n')
                            with open(file_name_COM_bin_count, 'w+') as fp:
                                fp.write('#COM_Bin\tAvg_bins_per_chain\n')
                                fp.write('#Order is chain type\n')
                            with open(file_name_COM_bead, 'w+') as fp:
                                fp.write('#COM_Bin\tbeads_per_bin\n')
                                fp.write('#Order is bead_type\n')
                            with open(file_name_COM_chain, 'w+') as fp:
                                fp.write('#COM_Bin\tnum_chains\tnum_beads\n')
                                fp.write('#Order is chain type\n')
                            with open(file_name_COM_sticker, 'w+') as fp:
                                fp.write('#COM_Bin\tfrac_satisfied_stickers\tavg_num_sticker_crosslinks\tfrac_inter_satisfied_stickers\tavg_num_inter_sticker_crosslinks\n')
                            with open(file_name_COM_Re, 'w+') as fp:
                                fp.write('#COM_Bin\t5_Bead_Re_squared\n')
                                fp.write('#Order is chain type\n')
                            with open(file_name_COM_angle, 'w+') as fp:
                                fp.write('#COM_Bin\tAvg_cosine_squared\n')
                                fp.write('#Order is chain type\n')
                            with open(file_name_COM_dot_prod, 'w+') as fp:
                                fp.write('#Chain_type\tChain_distance\tCOM_Bin\tAvg_dot_product\n')
                                fp.write('#Distance is based\n')
                            with open(file_name_COM_solvation, 'w+') as fp:
                                fp.write('#COM_Bin\tSolvated_sites\tSelf_sites\tOther_sites\tDegrees\n')
                                fp.write('#Order is chain type\n')
                        
                        if rdf_dist_check == 1:
                            with open(file_name_RDF, 'w+') as fp:
                                fp.write('#RDF_Bin\tRDF_Count\n')
    
                        if mol_com_dist_check == 1:
                            with open(file_name_mol_COM, 'w+') as fp:
                                fp.write('#Chain\tCluster_check\tx_COM\ty_COM\tz_COM\n')
                            with open(file_name_mol_COM_2, 'w+') as fp:
                                fp.write('#Chain\tDistance_Squared\n')
                            with open(file_name_mol_COM_3, 'w+') as fp:
                                fp.write('#Bin\tDistance_Squared\n')
                                fp.write('#Order is chain type\n')
                            with open(file_name_displacement, 'w+') as fp:
                                fp.write('#Bin\tChain_type\tDistance_Squared\n')
                            
                        if sticker_cluster_dist_check == 1:
                            with open(file_name_sticker_cluster, 'w+') as fp:
                                fp.write('#Cluster_size\tNum_clusters\n')
    
                        if crosslink_dist_check == 1:
                            with open(file_name_crosslink, 'w+') as fp:
                                fp.write('#Order is bead_type. Last column is solvation\n')
                                    
                        if solvation_dist_check == 1:
                            with open(file_name_solvation, 'w+') as fp:
                                fp.write('#Chain type\t#Rg squared\t#Empty sites\t#Self sites\t#Other sites\t#Degrees\n')
                                fp.write('#Order is whole chain then split into 4 parts\n')
                        
                        if com_gtensor_dist_check == 1:
                            with open(file_name_COM_gtensor, 'w+') as fp:
                                fp.write('#COM_Bin\tAsph\tAcyl\n')
                                fp.write('#Order is chain type\n')
                        
                        if connectedness_dist_check == 1:
                            with open(file_name_connectedness, 'w+') as fp:
                                fp.write('#Top 5% of connected chains per snapshot per chain type\n')
                                fp.write('#Chain\tBetweenness\tX-coord\tY-coord\tZ-coord\n')
                            with open(file_name_connectedness_binned, 'w+') as fp:
                                fp.write('#Betweenness bins are 0, 0-1, 1-5, 5-10, 10-50, 50-100, 100-500, 500-1000, 1000-5000, 5000+\n')
                                fp.write('#Betweenness_Bin\tMSD\n')
                                fp.write('#Order is chain type\n')
                            with open(file_name_connectedness_full, 'w+') as fp:
                                fp.write('#Betweenness\t#sticker interactions\tMSD\n')
                                fp.write('#Ordered by chain ID for each frame\n')
                        
                        if sticker_lifetime_dist_check == 1:
                            with open(file_name_sticker_lifetime, 'w+') as fp:
                                fp.write('#Number of frames that a sticker-sticker interactions existed\n')
                                fp.write('#Time between frames is %i MC moves\n' % time_inc)
    
                        if concave_hull_dist_check == 1:
                            with open(file_name_concave_hull, 'w+') as fp:
                                fp.write('#X,Y coordinates of points within the hull\n')
                                fp.write('#Slice thickness is %i and minimum beads at an X,Y slice is %i\n' % (int(concave_hull_ana_list[0]) * 2 + 1, int(concave_hull_ana_list[1])))
                                fp.write('#Time between frames is %i MC moves\n' % time_inc)
                        
                        with open(file_name_cluster, 'w+') as fp:
                            fp.write('#Cluster_size\tNum_clusters_dilute\tNum_cluster_dense\n')
    
                        for run_it in range(runs_per_condition):
                            
                            run_dir = Run_Path(chains_dir, run_it)
                            
                            if binary_check == 1:
                                file_list = [(run_dir + prefix + "_trj.lassi").encode('utf-8')]
                            else:
                                file_list = [(run_dir + prefix + "_trj.lammpstrj").encode('utf-8')]
                            
                            file_name_topo = run_dir + prefix + "_topo.lammpstrj"
                        
                            file_array = np.array(file_list)
                            array_view = file_array.view('uint8').reshape(file_array.size, file_array.dtype.itemsize)
                            
                            topo_array = np.array([file_name_topo.encode('utf-8')])
                            topo_array_view = topo_array.view('uint8').reshape(topo_array.size, topo_array.dtype.itemsize)
                            
                            nu_array = np.array([file_name_nu.encode('utf-8')])
                            nu_array_view = nu_array.view('uint8').reshape(nu_array.size, nu_array.dtype.itemsize)
    
                            rg_full_array = np.array([file_name_rg_full.encode('utf-8')])
                            rg_full_array_view = rg_full_array.view('uint8').reshape(rg_full_array.size, rg_full_array.dtype.itemsize)
            
                            re_dist_array = np.array([file_name_re_dist_clus.encode('utf-8'), file_name_re_dist_solo.encode('utf-8')])
                            re_dist_array_view = re_dist_array.view('uint8').reshape(re_dist_array.size, re_dist_array.dtype.itemsize)
                            
                            gtensor_dist_array = np.array([file_name_gtensor_dist_clus.encode('utf-8')])
                            gtensor_dist_array_view = gtensor_dist_array.view('uint8').reshape(gtensor_dist_array.size, gtensor_dist_array.dtype.itemsize)
            
                            fractal_ana_dist_array = np.array([file_name_fractal_ana_dist_clus.encode('utf-8')])
                            fractal_ana_dist_array_view = fractal_ana_dist_array.view('uint8').reshape(fractal_ana_dist_array.size, fractal_ana_dist_array.dtype.itemsize)
            
                            rg_COM_dist_array = np.array([file_name_rg_COM.encode('utf-8')])
                            rg_COM_dist_array_view = rg_COM_dist_array.view('uint8').reshape(rg_COM_dist_array.size, rg_COM_dist_array.dtype.itemsize)
            
                            COM_bin_count_dist_array = np.array([file_name_COM_bin_count.encode('utf-8')])
                            COM_bin_count_dist_array_view = COM_bin_count_dist_array.view('uint8').reshape(COM_bin_count_dist_array.size, COM_bin_count_dist_array.dtype.itemsize)
            
                            COM_bead_dist_array = np.array([file_name_COM_bead.encode('utf-8')])
                            COM_bead_dist_array_view = COM_bead_dist_array.view('uint8').reshape(COM_bead_dist_array.size, COM_bead_dist_array.dtype.itemsize)
            
                            COM_chain_dist_array = np.array([file_name_COM_chain.encode('utf-8')])
                            COM_chain_dist_array_view = COM_chain_dist_array.view('uint8').reshape(COM_chain_dist_array.size, COM_chain_dist_array.dtype.itemsize)
            
                            COM_sticker_dist_array = np.array([file_name_COM_sticker.encode('utf-8')])
                            COM_sticker_dist_array_view = COM_sticker_dist_array.view('uint8').reshape(COM_sticker_dist_array.size, COM_sticker_dist_array.dtype.itemsize)
            
                            COM_Re_dist_array = np.array([file_name_COM_Re.encode('utf-8')])
                            COM_Re_dist_array_view = COM_Re_dist_array.view('uint8').reshape(COM_Re_dist_array.size, COM_Re_dist_array.dtype.itemsize)
    
                            COM_angle_dist_array = np.array([file_name_COM_angle.encode('utf-8')])
                            COM_angle_dist_array_view = COM_angle_dist_array.view('uint8').reshape(COM_angle_dist_array.size, COM_angle_dist_array.dtype.itemsize)
    
                            COM_dot_prod_dist_array = np.array([file_name_COM_dot_prod.encode('utf-8')])
                            COM_dot_prod_dist_array_view = COM_dot_prod_dist_array.view('uint8').reshape(COM_dot_prod_dist_array.size, COM_dot_prod_dist_array.dtype.itemsize)
    
                            COM_solvation_dist_array = np.array([file_name_COM_solvation.encode('utf-8')])
                            COM_solvation_dist_array_view = COM_solvation_dist_array.view('uint8').reshape(COM_solvation_dist_array.size, COM_solvation_dist_array.dtype.itemsize)
            
                            RDF_dist_array = np.array([file_name_RDF.encode('utf-8')])
                            RDF_dist_array_view = RDF_dist_array.view('uint8').reshape(RDF_dist_array.size, RDF_dist_array.dtype.itemsize)
    
                            mol_COM_dist_array = np.array([file_name_mol_COM.encode('utf-8')])
                            mol_COM_dist_array_view = mol_COM_dist_array.view('uint8').reshape(mol_COM_dist_array.size, mol_COM_dist_array.dtype.itemsize)
    
                            mol_COM_dist_array_2 = np.array([file_name_mol_COM_2.encode('utf-8')])
                            mol_COM_dist_array_view_2 = mol_COM_dist_array_2.view('uint8').reshape(mol_COM_dist_array_2.size, mol_COM_dist_array_2.dtype.itemsize)
    
                            mol_COM_dist_array_3 = np.array([file_name_mol_COM_3.encode('utf-8')])
                            mol_COM_dist_array_view_3 = mol_COM_dist_array_3.view('uint8').reshape(mol_COM_dist_array_3.size, mol_COM_dist_array_3.dtype.itemsize)
    
                            displacement_dist_array = np.array([file_name_displacement.encode('utf-8')])
                            displacement_dist_array_view = displacement_dist_array.view('uint8').reshape(displacement_dist_array.size, displacement_dist_array.dtype.itemsize)
    
                            sticker_cluster_dist_array = np.array([file_name_sticker_cluster.encode('utf-8')])
                            sticker_cluster_dist_array_view = sticker_cluster_dist_array.view('uint8').reshape(sticker_cluster_dist_array.size, sticker_cluster_dist_array.dtype.itemsize)
    
                            crosslink_dist_array = np.array([file_name_crosslink.encode('utf-8')])
                            crosslink_dist_array_view = crosslink_dist_array.view('uint8').reshape(crosslink_dist_array.size, crosslink_dist_array.dtype.itemsize)
                            
                            cluster_dist_array = np.array([file_name_cluster.encode('utf-8')])
                            cluster_dist_array_view = cluster_dist_array.view('uint8').reshape(cluster_dist_array.size, cluster_dist_array.dtype.itemsize)
    
                            solvation_dist_array = np.array([file_name_solvation.encode('utf-8')])
                            solvation_dist_array_view = solvation_dist_array.view('uint8').reshape(solvation_dist_array.size, solvation_dist_array.dtype.itemsize)
    
                            COM_gtensor_dist_array = np.array([file_name_COM_gtensor.encode('utf-8')])
                            COM_gtensor_dist_array_view = COM_gtensor_dist_array.view('uint8').reshape(COM_gtensor_dist_array.size, COM_gtensor_dist_array.dtype.itemsize)
    
                            COM_gtensor_dist_array = np.array([file_name_COM_gtensor.encode('utf-8')])
                            COM_gtensor_dist_array_view = COM_gtensor_dist_array.view('uint8').reshape(COM_gtensor_dist_array.size, COM_gtensor_dist_array.dtype.itemsize)
    
                            connectedness_dist_array = np.array([file_name_connectedness.encode('utf-8')])
                            connectedness_dist_array_view = connectedness_dist_array.view('uint8').reshape(connectedness_dist_array.size, connectedness_dist_array.dtype.itemsize)
    
                            connectedness_dist_binned_array = np.array([file_name_connectedness_binned.encode('utf-8')])
                            connectedness_dist_binned_array_view = connectedness_dist_binned_array.view('uint8').reshape(connectedness_dist_binned_array.size, connectedness_dist_binned_array.dtype.itemsize)
    
                            connectedness_dist_full_array = np.array([file_name_connectedness_full.encode('utf-8')])
                            connectedness_dist_full_array_view = connectedness_dist_full_array.view('uint8').reshape(connectedness_dist_full_array.size, connectedness_dist_full_array.dtype.itemsize)
    
                            sticker_lifetime_dist_full_array = np.array([file_name_sticker_lifetime.encode('utf-8')])
                            sticker_lifetime_dist_full_array_view = sticker_lifetime_dist_full_array.view('uint8').reshape(sticker_lifetime_dist_full_array.size, sticker_lifetime_dist_full_array.dtype.itemsize)
    
                            concave_hull_dist_full_array = np.array([file_name_concave_hull.encode('utf-8')])
                            concave_hull_dist_full_array_view = concave_hull_dist_full_array.view('uint8').reshape(concave_hull_dist_full_array.size, concave_hull_dist_full_array.dtype.itemsize)
                            
                            Sticker_array = np.array(sticker_list, dtype='int')
                            Sticker_array_length = len(sticker_list)
                            
                            traj_analyzer_multi_new.Frame_Loader(array_view.shape[0], binary_check, array_view, array_view.shape[1],
                                                                 topo_array_view, topo_array_view.shape[1],
                                                                 nu_array_view, nu_array_view.shape[1], rg_full_array_view,
                                                                 rg_full_array_view.shape[1], re_dist_array_view,
                                                                 re_dist_array_view.shape[1], gtensor_dist_array_view,
                                                                 gtensor_dist_array_view.shape[1], fractal_ana_dist_array_view,
                                                                 fractal_ana_dist_array_view.shape[1], rg_COM_dist_array_view,
                                                                 rg_COM_dist_array_view.shape[1], COM_bin_count_dist_array_view,
                                                                 COM_bin_count_dist_array_view.shape[1], COM_bead_dist_array_view,
                                                                 COM_bead_dist_array_view.shape[1], COM_chain_dist_array_view,
                                                                 COM_chain_dist_array_view.shape[1], COM_sticker_dist_array_view,
                                                                 COM_sticker_dist_array_view.shape[1], COM_Re_dist_array_view,
                                                                 COM_Re_dist_array_view.shape[1], COM_angle_dist_array_view,
                                                                 COM_angle_dist_array_view.shape[1], COM_dot_prod_dist_array_view,
                                                                 COM_dot_prod_dist_array_view.shape[1], COM_solvation_dist_array_view,
                                                                 COM_solvation_dist_array_view.shape[1], RDF_dist_array_view,
                                                                 RDF_dist_array_view.shape[1], mol_COM_dist_array_view,
                                                                 mol_COM_dist_array_view.shape[1], mol_COM_dist_array_view_2,
                                                                 mol_COM_dist_array_view_2.shape[1], mol_COM_dist_array_view_3,
                                                                 mol_COM_dist_array_view_3.shape[1], displacement_dist_array_view,
                                                                 displacement_dist_array_view.shape[1], sticker_cluster_dist_array_view,
                                                                 sticker_cluster_dist_array_view.shape[1], crosslink_dist_array_view,
                                                                 crosslink_dist_array_view.shape[1], solvation_dist_array_view,
                                                                 solvation_dist_array_view.shape[1], COM_gtensor_dist_array_view,
                                                                 COM_gtensor_dist_array_view.shape[1], connectedness_dist_array_view,
                                                                 connectedness_dist_array_view.shape[1], connectedness_dist_binned_array_view,
                                                                 connectedness_dist_binned_array_view.shape[1], connectedness_dist_full_array_view,
                                                                 connectedness_dist_full_array_view.shape[1], sticker_lifetime_dist_full_array_view,
                                                                 sticker_lifetime_dist_full_array_view.shape[1], concave_hull_dist_full_array_view,
                                                                 concave_hull_dist_full_array_view.shape[1], cluster_dist_array_view,
                                                                 cluster_dist_array_view.shape[1], time_start, time_inc, time_end,
                                                                 time_cutoff, int(nu_check), int(re_dist_check), int(gtensor_dist_check),
                                                                 int(fractal_ana_dist_check), int(com_dist_check), int(rdf_dist_check),
                                                                 int(mol_com_dist_check), int(sticker_cluster_dist_check),
                                                                 int(crosslink_dist_check), int(solvation_dist_check),
                                                                 int(com_gtensor_dist_check), int(connectedness_dist_check),
                                                                 int(sticker_lifetime_dist_check), int(concave_hull_dist_check),
                                                                 Sticker_array, Sticker_array_length,
                                                                 fractal_ana_list[0], fractal_ana_list[1], fractal_ana_list[2],
                                                                 int(concave_hull_ana_list[0]), int(concave_hull_ana_list[1]),
                                                                 int(cur_dist_chains), cur_chains_array)
    
                            if nu_check == 1:
                                with open(file_name_nu, 'a') as fp:
                                    fp.write('#NewRun\n')
                                with open(file_name_rg_full, 'a') as fp:
                                    fp.write('#NewRun\n')
            
                            if com_dist_check == 1:
                                with open(file_name_rg_COM, 'a') as fp:
                                    fp.write('#NewRun\n')
                        
                                with open(file_name_COM_bin_count, 'a') as fp:
                                    fp.write('#NewRun\n')
            
                                with open(file_name_COM_bead, 'a') as fp:
                                    fp.write('#NewRun\n')
            
                                with open(file_name_COM_chain, 'a') as fp:
                                    fp.write('#NewRun\n')
                
                                with open(file_name_COM_sticker, 'a') as fp:
                                    fp.write('#NewRun\n')
            
                                with open(file_name_COM_Re, 'a') as fp:
                                    fp.write('#NewRun\n')
    
                                with open(file_name_COM_angle, 'a') as fp:
                                    fp.write('#NewRun\n')
    
                                with open(file_name_COM_dot_prod, 'a') as fp:
                                    fp.write('#NewRun\n')
    
                                with open(file_name_COM_solvation, 'a') as fp:
                                    fp.write('#NewRun\n')
            
                            if rdf_dist_check == 1:
                                with open(file_name_RDF, 'a') as fp:
                                    fp.write('#NewRun\n')
    
                            if mol_com_dist_check == 1:
                                with open(file_name_mol_COM, 'a') as fp:
                                    fp.write('#NewRun\n')
                                with open(file_name_mol_COM_2, 'a') as fp:
                                    fp.write('#NewRun\n')
                                with open(file_name_mol_COM_3, 'a') as fp:
                                    fp.write('#NewRun\n')
            
                            if sticker_cluster_dist_check == 1:
                                with open(file_name_sticker_cluster, 'a') as fp:
                                    fp.write('#NewRun\n')
    
                            if crosslink_dist_check == 1:
                                with open(file_name_crosslink, 'a') as fp:
                                    fp.write('#NewRun\n')
    
                            if solvation_dist_check == 1:
                                with open(file_name_solvation, 'a') as fp:
                                    fp.write('#NewRun\n')
    
                            if com_gtensor_dist_check == 1:
                                with open(file_name_COM_gtensor, 'a') as fp:
                                    fp.write('#NewRun\n')
                                    
                            if connectedness_dist_check == 1:
                                with open(file_name_connectedness, 'a') as fp:
                                    fp.write('#NewRun\n')
                                with open(file_name_connectedness_binned, 'a') as fp:
                                    fp.write('#NewRun\n')
                                with open(file_name_connectedness_full, 'a') as fp:
                                    fp.write('#NewRun\n')
    
                            if sticker_lifetime_dist_check == 1:
                                with open(file_name_sticker_lifetime, 'a') as fp:
                                    fp.write('#NewRun\n')
    
                            if concave_hull_dist_check == 1:
                                with open(file_name_concave_hull, 'a') as fp:
                                    fp.write('#NewRun\n')
    
                            with open(file_name_cluster, 'a') as fp:
                                fp.write('#NewRun\n')
                    
                    if concave_hull_ana_check == 1:

                        with open(file_name_concave_hull_fft, 'w+') as fp:
                            fp.write('#Fast Fourier Transform of concave hull distribution\n')
                            fp.write('#Each line is an X,Y coordinate\n')
                            fp.write('#Slice thickness is %i and minimum beads at an X,Y slice is %i\n' % (int(concave_hull_ana_list[0]) * 2 + 1, int(concave_hull_ana_list[1])))
                            fp.write('#alphashape parameter is %.2f and number of theta values is %i\n' % (float(concave_hull_ana_list[2]), int(concave_hull_ana_list[3])))
                            fp.write('#Time between frames is %i MC moves\n' % (time_inc * int(concave_hull_ana_list[4])))
                                                        
                        concave_hull_ana.Concave_Hull_Analysis(file_name_concave_hull, file_name_concave_hull_fft, concave_hull_ana_list[2], int(concave_hull_ana_list[3]), int(concave_hull_ana_list[4]))
                    
                    if mol_com_ana_check == 1:
                        
                        mol_COM_dist_array = np.array([file_name_mol_COM.encode('utf-8')])
                        mol_COM_dist_array_view = mol_COM_dist_array.view('uint8').reshape(mol_COM_dist_array.size, mol_COM_dist_array.dtype.itemsize)

                        with open(file_name_mol_COM_ana, 'w+') as fp:
                            fp.write('#Mean square displacements of dense phase chain centers-of-mass\n')
                            fp.write('#Ordered by time increment\n')
                            fp.write('#Base increment is %i MC moves\n' % time_inc)
                        
                        mol_COM_dist_array_ana = np.array([file_name_mol_COM_ana.encode('utf-8')])
                        mol_COM_dist_array_view_ana = mol_COM_dist_array_ana.view('uint8').reshape(mol_COM_dist_array_ana.size, mol_COM_dist_array_ana.dtype.itemsize)
                        num_frames = int((time_end - time_cutoff) / time_inc + 1)
                        print(num_chains)
                        print(num_frames)
                        print(box)
                        msd_ana.MSD_Analysis(mol_COM_dist_array_view, mol_COM_dist_array_view.shape[1],
                                             mol_COM_dist_array_view_ana, mol_COM_dist_array_view_ana.shape[1],
                                             num_chains, num_frames, int(box))







