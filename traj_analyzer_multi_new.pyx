'''
Created on Apr 26, 2020

@author: mina
'''
### To compile, use command "python cython_setup.py build_ext --inplace"
DEF buffer_size = 100000
DEF RE_DIST_ARRAY_SIZE = 10000
DEF GYRATION_TENSOR_DIST_ARRAY_SIZE = 1000
DEF FRACTAL_ANA_DIST_ARRAY_SIZE = 2000
DEF FRACTAL_SOLVATION_ANA_DIST_ARRAY_SIZE = 2000
DEF SOLVATION_ANA_DIST_ARRAY_SIZE = 1000
DEF DISPLACEMENT_ANA_DIST_ARRAY_SIZE = 1000
DEF CONNECTEDNESS_ANA_DIST_ARRAY_SIZE = 200
DEF CONNECTEDNESS_FULL_ARRAY_SIZE = 50
DEF STICKER_LIFETIME_ARRAY_SIZE = 10000
DEF DISPLACEMENT_SKIP = 23
DEF CONNECTEDNESS_SKIP = 0

DEF FRAME_ARRAY_X = 0
DEF FRAME_ARRAY_Y = 1
DEF FRAME_ARRAY_Z = 2
DEF FRAME_ARRAY_BP = 3
DEF FRAME_ARRAY_MAX = 4

DEF TOPO_ARRAY_CHAIN = 0
DEF TOPO_ARRAY_BEAD = 1
DEF TOPO_ARRAY_MAX = 2

DEF CROSSLINK_CLUS_ST_ST = 0
DEF CROSSLINK_CLUS_ST_SP = 1
DEF CROSSLINK_CLUS_ST_SO = 2
DEF CROSSLINK_CLUS_SP_ST = 3
DEF CROSSLINK_CLUS_SP_SP = 4
DEF CROSSLINK_CLUS_SP_SO = 5
DEF CROSSLINK_DIL_ST_ST = 6
DEF CROSSLINK_DIL_ST_SP = 7
DEF CROSSLINK_DIL_ST_SO = 8
DEF CROSSLINK_DIL_SP_ST = 9
DEF CROSSLINK_DIL_SP_SP = 10
DEF CROSSLINK_DIL_SP_SO = 11
DEF CROSSLINK_CLUS_ST_ST_CHAIN = 12
DEF CROSSLINK_CLUS_ST_SP_CHAIN = 13
DEF CROSSLINK_CLUS_SP_ST_CHAIN = 14
DEF CROSSLINK_CLUS_SP_SP_CHAIN = 15
DEF CROSSLINK_CLUS_ST_ST_CON = 16
DEF CROSSLINK_CLUS_ST_SP_CON = 17
DEF CROSSLINK_CLUS_SP_ST_CON = 18
DEF CROSSLINK_CLUS_SP_SP_CON = 19
DEF CROSSLINK_MAX = 20

DEF CROSSLINK_COUNT_CLUS_ST = 0
DEF CROSSLINK_COUNT_CLUS_SP = 1
DEF CROSSLINK_COUNT_DIL_ST = 2
DEF CROSSLINK_COUNT_DIL_SP = 3
DEF CROSSLINK_COUNT_MAX = 4

DEF CROSSLINK_FULL_CLUS = 0
DEF CROSSLINK_FULL_DIL = 1
DEF CROSSLINK_FULL_CLUS_CHAIN = 2
DEF CROSSLINK_FULL_DIL_CHAIN = 3
DEF CROSSLINK_FULL_MAX = 4

DEF CROSSLINK_FULL_COUNT_CLUS = 0
DEF CROSSLINK_FULL_COUNT_DIL = 1
DEF CROSSLINK_FULL_COUNT_MAX = 2

DEF COM_SOLVATION_EMPTY = 0
DEF COM_SOLVATION_SELF = 1
DEF COM_SOLVATION_OTHER = 2
DEF COM_SOLVATION_DEGREES = 3
DEF COM_SOLVATION_MAX = 4

DEF FRACTAL_X = 0
DEF FRACTAL_Y = 1
DEF FRACTAL_Z = 2
DEF FRACTAL_DISTANCE = 3
DEF FRACTAL_EMPTY = 4
DEF FRACTAL_STICKER = 5
DEF FRACTAL_EMPTY_NEIGHBOR = 6
DEF FRACTAL_STICKER_NEIGHBOR = 7
DEF FRACTAL_SPACER_NEIGHBOR = 8
DEF FRACTAL_MAX = 9

DEF PARTIAL_RG_NUM = 4

DEF SOLVATION_RG = 0
DEF SOLVATION_EMPTY_SITE = 1
DEF SOLVATION_SELF_SITE = 2
DEF SOLVATION_CHAIN_SITE = 3
DEF SOLVATION_DEGREES = 4
DEF SOLVATION_MAX_NUM = 5
DEF SOLVATION_MAX = SOLVATION_MAX_NUM * (PARTIAL_RG_NUM + 1) + 1
DEF SOLVATION_CHAIN_TYPE = SOLVATION_MAX - 1

DEF CONNECTEDNESS_CHAIN = 0
DEF CONNECTEDNESS_DEGREES = 1
DEF CONNECTEDNESS_X = 2
DEF CONNECTEDNESS_Y = 3
DEF CONNECTEDNESS_Z = 4
DEF CONNECTEDNESS_MAX = 5

DEF RG_FULL_DIL = 0
DEF RG_FULL_DEN = 1
DEF RG_FULL_MAX = 2

DEF BETWEENNESS_BINS = 10

cimport cython
import numpy as np
cimport numpy as np
np.import_array()
import betweenness_calculator
from libc.math cimport sqrt, log, abs
from libc.stdio cimport fopen, fclose, FILE, EOF, fseek, SEEK_END, SEEK_CUR, SEEK_SET
from libc.stdio cimport ftell, fgetc, fgets, getc, gets, feof, fread, getline
from libc.stdio cimport sscanf, fprintf
from libc.stdlib cimport calloc, malloc, free, atof, rand, RAND_MAX

cdef:
    int file_num[1]
    int nu_file_size[1]
    int rg_full_file_size[1]
    int re_dist_file_size[1]
    int gtensor_dist_file_size[1]
    int fractal_ana_dist_file_size[1]
    int rg_COM_dist_file_size[1]
    int COM_bin_count_dist_file_size[1]
    int COM_bead_dist_file_size[1]
    int COM_chain_dist_file_size[1]
    int COM_sticker_dist_file_size[1]
    int COM_Re_dist_file_size[1]
    int COM_angle_dist_file_size[1]
    int COM_dot_prod_dist_file_size[1]
    int COM_solvation_dist_file_size[1]
    int RDF_dist_file_size[1]
    int mol_COM_dist_file_size[1]
    int mol_COM_dist_file_size_2[1]
    int mol_COM_dist_file_size_3[1]
    int displacement_dist_file_size[1]
    int sticker_cluster_dist_file_size[1]
    int crosslink_dist_file_size[1]
    int cluster_dist_file_size[1]
    int solvation_dist_file_size[1]
    int COM_gtensor_dist_file_size[1]
    int connectedness_dist_file_size[1]
    int connectedness_binned_dist_file_size[1]
    int connectedness_full_dist_file_size[1]
    int sticker_lifetime_dist_file_size[1]
    int concave_hull_dist_file_size[1]
    int num_distinct_chains[1]
    int total_num_chains[1]
    int chain_lengths_sum[1]
    int total_scaling_length[1]
    int num_beads[1]
    int box_size[1]
    int bead_types[1]
    int ana_frames[1]
    int count_analysis[1]
    int re_count_clus[1]
    int re_count_solo[1]
    int gtensor_count[1]
    int fractal_ana_count[1]
    int COM_chain_count[1]
    int solvation_count[1]
    int displacement_count[1]
    int displacement_skip_count[1]
    int connectedness_count[1]
    int connectedness_full_count[1]
    int sticker_lifetime_count[1]
    int nu_check[1]
    int re_dist_check[1]
    int gtensor_dist_check[1]
    int fractal_ana_dist_check[1]
    int com_dist_check[1]
    int RDF_dist_check[1]
    int mol_COM_dist_check[1]
    int sticker_cluster_dist_check[1]
    int crosslink_dist_check[1]
    int solvation_dist_check[1]
    int com_gtensor_dist_check[1]
    int connectedness_dist_check[1]
    int sticker_lifetime_dist_check[1]
    int concave_hull_dist_check[1]
    int num_stickers[1]
    int total_num_stickers[1]
    int fractal_ana_samples[1]
    int fractal_ana_radius[1]
    int fractal_ana_max_distance[1]
    int connectedness_chains[1]
    int connectedness_skip_count[1]
    int concave_hull_z_thickness[1]
    int concave_hull_z_requirement[1]
    long total_num_cluster_stickers[1]
    int crosslink_conversion_array[CROSSLINK_MAX]
    int crosslink_full_conversion_array[CROSSLINK_FULL_MAX]
    int sticker_lifetime_dist_array[STICKER_LIFETIME_ARRAY_SIZE]
    long crosslink_ana_array[CROSSLINK_MAX]
    long crosslink_count_array[CROSSLINK_COUNT_MAX]
    long double re_dist_array_clus[2 * RE_DIST_ARRAY_SIZE]
    long double re_dist_array_solo[2 * RE_DIST_ARRAY_SIZE]
    long double gtensor_dist_array[6 * GYRATION_TENSOR_DIST_ARRAY_SIZE]
    long double fractal_ana_dist_array[FRACTAL_MAX * FRACTAL_ANA_DIST_ARRAY_SIZE]
    long double solvation_array[SOLVATION_MAX * SOLVATION_ANA_DIST_ARRAY_SIZE]
    long double displacement_array[3 * DISPLACEMENT_ANA_DIST_ARRAY_SIZE]
    long double nu_mean[1]
    long double nu_err[1]

crosslink_conversion_array[CROSSLINK_CLUS_ST_ST] = CROSSLINK_COUNT_CLUS_ST
crosslink_conversion_array[CROSSLINK_CLUS_ST_SP] = CROSSLINK_COUNT_CLUS_ST
crosslink_conversion_array[CROSSLINK_CLUS_ST_SO] = CROSSLINK_COUNT_CLUS_ST
crosslink_conversion_array[CROSSLINK_CLUS_SP_ST] = CROSSLINK_COUNT_CLUS_SP
crosslink_conversion_array[CROSSLINK_CLUS_SP_SP] = CROSSLINK_COUNT_CLUS_SP
crosslink_conversion_array[CROSSLINK_CLUS_SP_SO] = CROSSLINK_COUNT_CLUS_SP
crosslink_conversion_array[CROSSLINK_DIL_ST_ST] = CROSSLINK_COUNT_DIL_ST
crosslink_conversion_array[CROSSLINK_DIL_ST_SP] = CROSSLINK_COUNT_DIL_ST
crosslink_conversion_array[CROSSLINK_DIL_ST_SO] = CROSSLINK_COUNT_DIL_ST
crosslink_conversion_array[CROSSLINK_DIL_SP_ST] = CROSSLINK_COUNT_DIL_SP
crosslink_conversion_array[CROSSLINK_DIL_SP_SP] = CROSSLINK_COUNT_DIL_SP
crosslink_conversion_array[CROSSLINK_DIL_SP_SO] = CROSSLINK_COUNT_DIL_SP
crosslink_conversion_array[CROSSLINK_CLUS_ST_ST_CHAIN] = CROSSLINK_COUNT_CLUS_ST
crosslink_conversion_array[CROSSLINK_CLUS_ST_SP_CHAIN] = CROSSLINK_COUNT_CLUS_ST
crosslink_conversion_array[CROSSLINK_CLUS_SP_ST_CHAIN] = CROSSLINK_COUNT_CLUS_SP
crosslink_conversion_array[CROSSLINK_CLUS_SP_SP_CHAIN] = CROSSLINK_COUNT_CLUS_SP
crosslink_conversion_array[CROSSLINK_CLUS_ST_ST_CON] = CROSSLINK_COUNT_CLUS_ST
crosslink_conversion_array[CROSSLINK_CLUS_ST_SP_CON] = CROSSLINK_COUNT_CLUS_ST
crosslink_conversion_array[CROSSLINK_CLUS_SP_ST_CON] = CROSSLINK_COUNT_CLUS_SP
crosslink_conversion_array[CROSSLINK_CLUS_SP_SP_CON] = CROSSLINK_COUNT_CLUS_SP

crosslink_full_conversion_array[CROSSLINK_FULL_CLUS] = CROSSLINK_FULL_COUNT_CLUS
crosslink_full_conversion_array[CROSSLINK_FULL_CLUS_CHAIN] = CROSSLINK_FULL_COUNT_CLUS
crosslink_full_conversion_array[CROSSLINK_FULL_DIL] = CROSSLINK_FULL_COUNT_DIL
crosslink_full_conversion_array[CROSSLINK_FULL_DIL_CHAIN] = CROSSLINK_FULL_COUNT_DIL

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cdef long Dist_Bead_Bead_1D_No_PBC(int *frame_array, const int bead1, const int bead2, const int dimension):
    cdef:
        int dist
    dist = abs(frame_array[FRAME_ARRAY_MAX * bead1 + dimension] - frame_array[FRAME_ARRAY_MAX * bead2 + dimension])
    
    return dist

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cdef long Dist_Bead_Bead_1D(int *frame_array, const int bead1, const int bead2, const int dimension):
    cdef:
        int dist
    dist = abs(frame_array[FRAME_ARRAY_MAX * bead1 + dimension] - frame_array[FRAME_ARRAY_MAX * bead2 + dimension])
    if dist > (box_size[0] / 2):
        dist = box_size[0] - dist
     
    return dist

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
#Currently has PBC
cdef long Dist_Bead_Bead_Squared_No_PBC(int *frame_array, const int bead1, const int bead2):
    cdef:
        int i
        int dist_array[3]
    for i in range(3):
        dist_array[i] = abs(frame_array[FRAME_ARRAY_MAX * bead1 + i] - frame_array[FRAME_ARRAY_MAX * bead2 + i])
        if dist_array[i] > (box_size[0] / 2):
            dist_array[i] = box_size[0] - dist_array[i]
    
    return (dist_array[0] * dist_array[0] + dist_array[1] * dist_array[1] + dist_array[2] * dist_array[2])

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cdef long Dist_Bead_Bead_Squared(int *frame_array, const int bead1, const int bead2):
    cdef:
        int i
        int dist_array[3]
    for i in range(3):
        dist_array[i] = abs(frame_array[FRAME_ARRAY_MAX * bead1 + i] - frame_array[FRAME_ARRAY_MAX * bead2 + i])
        if dist_array[i] > (box_size[0] / 2):
            dist_array[i] = box_size[0] - dist_array[i]
    
    return (dist_array[0] * dist_array[0] + dist_array[1] * dist_array[1] + dist_array[2] * dist_array[2])

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cdef long Dist_Bead_Point_Squared(int *frame_array, const int bead1, int *point_array):
    cdef:
        int i
        int dist_array[3]
    for i in range(3):
        dist_array[i] = abs(frame_array[FRAME_ARRAY_MAX * bead1 + i] - point_array[i])
        if dist_array[i] > (box_size[0] / 2):
            dist_array[i] = box_size[0] - dist_array[i]
    
    return (dist_array[0] * dist_array[0] + dist_array[1] * dist_array[1] + dist_array[2] * dist_array[2])

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cdef long Dist_Bead_Float_Point_Squared(int *frame_array, const int bead1, long double *point_array):
    cdef:
        int i
        int dist_array[3]
    for i in range(3):
        dist_array[i] = abs(frame_array[FRAME_ARRAY_MAX * bead1 + i] - point_array[i])
        if dist_array[i] > (box_size[0] / 2):
            dist_array[i] = box_size[0] - dist_array[i]
    
    return (dist_array[0] * dist_array[0] + dist_array[1] * dist_array[1] + dist_array[2] * dist_array[2])

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cdef long double Cos_Angle_Two_Beads_Point(int *frame_array, const int bead1, const int bead2, int *point_array):
    cdef:
        long double bead1_bead2_dist
        long double bead1_center_dist
        long double bead2_center_dist
    
    bead1_bead2_dist = Dist_Bead_Bead_Squared(frame_array, bead1, bead2)
    bead1_center_dist = Dist_Bead_Point_Squared(frame_array, bead1, point_array)
    bead2_center_dist = Dist_Bead_Point_Squared(frame_array, bead2, point_array)
    #check if the first bead is at the system center
    if bead1_center_dist == 0:
        #we return an impossible value so we can catch it later
        return 5
    return <long double> (bead1_bead2_dist + bead1_center_dist - bead2_center_dist) / <long double> (2 * sqrt(bead1_bead2_dist) * sqrt(bead1_center_dist))

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cdef int Rotation_Check(int *lattice_array, const int *tmpR1, const int rotation):
    
    cdef:
        int i
        int tmpR2[3]
        int lattice_loc = 0
    
    if rotation < 9:
        tmpR2[0] = -1
    elif rotation < 18:
        tmpR2[0] = 0
    else:
        tmpR2[0] = 1
    
    if rotation % 9 < 3:
        tmpR2[1] = -1
    elif rotation % 9 < 6:
        tmpR2[1] = 0
    else:
        tmpR2[1] = 1
    
    tmpR2[2] = (rotation % 3) - 1
    
    for i in range(3):
        tmpR2[i] += tmpR1[i]
        tmpR2[i] = tmpR2[i] % box_size[0]
    
    lattice_loc += tmpR2[0] * box_size[0] * box_size[0]
    lattice_loc += tmpR2[1] * box_size[0]
    lattice_loc += tmpR2[2]
    
    return lattice_array[lattice_loc]

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cdef COM_Bead_Writer(char* COM_bead_file, long *COM_bead_array):
     
    cdef:
        int i, j
        long double cur_bin_bead
        FILE* f_COM_bead
     
    f_COM_bead = fopen(COM_bead_file[:COM_bead_dist_file_size[0]], "a")
    for i in range(4 * box_size[0]):
        fprintf(f_COM_bead, "%d", i)
        for j in range(bead_types[0]):
            cur_bin_bead = <long double> COM_bead_array[bead_types[0] * i + j] / <long double> count_analysis[0]
            fprintf(f_COM_bead, "\t%LE", cur_bin_bead)
        fprintf(f_COM_bead, "\n")
    fclose(f_COM_bead)

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cdef COM_Chain_Writer(char* COM_chain_file, long *COM_chain_array):
     
    cdef:
        int i, j
        long double bin_beads
        long double bin_chains
        FILE* f_COM_chain
     
    f_COM_chain = fopen(COM_chain_file[:COM_chain_dist_file_size[0]], "a")
    for i in range(4 * box_size[0]):
        fprintf(f_COM_chain, "%d\t", i)
        for j in range(num_distinct_chains[0]):
            bin_chains = <long double> COM_chain_array[2 * i * num_distinct_chains[0] + 2 * j] / <long double> COM_chain_count[0]
            bin_beads = <long double> COM_chain_array[2 * i * num_distinct_chains[0] + 2 * j + 1] / <long double> COM_chain_count[0]
            fprintf(f_COM_chain, "%LE\t%LE\t", bin_chains, bin_beads)
        fprintf(f_COM_chain, "\n")
    fclose(f_COM_chain)

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cdef COM_Sticker_Writer(char* COM_sticker_file, long *COM_sticker_array,
                        long *COM_sticker_count_array):
     
    cdef:
        int i
        long double num_stickers_binary, num_inter_stickers_binary
        long double num_stickers_total, num_inter_stickers_total
        FILE* f_COM_sticker
     
    f_COM_sticker = fopen(COM_sticker_file[:COM_sticker_dist_file_size[0]], "a")
    for i in range(4 * box_size[0]):
        if COM_sticker_count_array[i] == 0:
            fprintf(f_COM_sticker, "%d\t-1\t-1\t-1\t-1\n", i)
        else:
            num_stickers_binary = <long double> COM_sticker_array[4 * i] / <long double> COM_sticker_count_array[i]
            num_stickers_total = <long double> COM_sticker_array[4 * i + 1] / <long double> COM_sticker_count_array[i]
            num_inter_stickers_binary = <long double> COM_sticker_array[4 * i + 2] / <long double> COM_sticker_count_array[i]
            num_inter_stickers_total = <long double> COM_sticker_array[4 * i + 3] / <long double> COM_sticker_count_array[i]
            fprintf(f_COM_sticker, "%d\t%LE\t%LE\t%LE\t%LE\n", i, num_stickers_binary, num_stickers_total, num_inter_stickers_binary, num_inter_stickers_total)
    fclose(f_COM_sticker)

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cdef COM_Re_Writer(char* COM_Re_file, long double *COM_Re_array, long *COM_Re_count_array):
     
    cdef:
        int i, j
        FILE* f_COM_Re
     
    f_COM_Re = fopen(COM_Re_file[:COM_Re_dist_file_size[0]], "a")
    for i in range(4 * box_size[0]):
        fprintf(f_COM_Re, "%d\t", i)
        for j in range(num_distinct_chains[0]):
            if COM_Re_count_array[i * num_distinct_chains[0] + j] == 0:
                fprintf(f_COM_Re, "%d\t", 0)
            else:
                fprintf(f_COM_Re, "%LE\t", (<long double> COM_Re_array[i * num_distinct_chains[0] + j] / <long double> COM_Re_count_array[i * num_distinct_chains[0] + j]))
        fprintf(f_COM_Re, "\n")
    fclose(f_COM_Re)

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cdef COM_Angle_Writer(char* COM_angle_file, long double *COM_angle_array,
                      long *COM_angle_count_array):
     
    cdef:
        int i, j
        FILE* f_COM_angle
     
    f_COM_angle = fopen(COM_angle_file[:COM_angle_dist_file_size[0]], "a")
    for i in range(4 * box_size[0]):
        fprintf(f_COM_angle, "%d\t", i)
        for j in range(num_distinct_chains[0]):
            if COM_angle_count_array[i * num_distinct_chains[0] + j] == 0:
                fprintf(f_COM_angle, "%d\t", 0)
            else:
                fprintf(f_COM_angle, "%LE\t", (<long double> COM_angle_array[i * num_distinct_chains[0] + j] / <long double> COM_angle_count_array[i * num_distinct_chains[0] + j]))
        fprintf(f_COM_angle, "\n")
    fclose(f_COM_angle)

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cdef COM_Dot_Prod_Writer(char* COM_dot_prod_file, long double *COM_dot_prod_array,
                         long *COM_dot_prod_count_array, const int *chain_length_fb_array,
                         const int *chain_length_array):
                
    cdef:
        int i, j, k, cur_dist
        FILE* f_COM_dot_prod
        
    f_COM_dot_prod = fopen(COM_dot_prod_file[:COM_dot_prod_dist_file_size[0]], "a")
    for i in range(num_distinct_chains[0]):
        for j in range(chain_length_fb_array[i], chain_length_fb_array[i] + chain_length_array[i]):
            cur_dist = j - chain_length_fb_array[i]
            for k in range(4 * box_size[0]):
                if j == chain_length_fb_array[i]:
                    fprintf(f_COM_dot_prod, "%d\t%d\t%d\t%d\n", i, cur_dist, k, 1)
                elif COM_dot_prod_count_array[j * 4 * box_size[0] + k] == 0:
                    fprintf(f_COM_dot_prod, "%d\t%d\t%d\t%d\n", i, cur_dist, k, 10)
                else:
                    fprintf(f_COM_dot_prod, "%d\t%d\t%d\t%LE\n", i, cur_dist, k, (<long double> COM_dot_prod_array[j * 4 * box_size[0] + k] / <long double> COM_dot_prod_count_array[j * 4 * box_size[0] + k]))
    fclose(f_COM_dot_prod)

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cdef COM_Solvation_Writer(char* COM_solvation_file, long double *COM_solvation_array,
                          long *COM_solvation_count_array):
     
    cdef:
        int i, j, k
        FILE* f_COM_solvation
     
    f_COM_solvation = fopen(COM_solvation_file[:COM_solvation_dist_file_size[0]], "a")
    for i in range(4 * box_size[0]):
        fprintf(f_COM_solvation, "%d\t", i)
        for j in range(num_distinct_chains[0]):
            for k in range(COM_SOLVATION_MAX):
                if COM_solvation_count_array[i * num_distinct_chains[0] + j] == 0:
                    fprintf(f_COM_solvation, "%d\t", 0)
                else:
                    fprintf(f_COM_solvation, "%LE\t", (<long double > COM_solvation_array[i * num_distinct_chains[0] * COM_SOLVATION_MAX + j * COM_SOLVATION_MAX + k] / <long double> COM_solvation_count_array[i * num_distinct_chains[0] + j]))
        fprintf(f_COM_solvation, "\n")
    fclose(f_COM_solvation)

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cdef COM_GTensor_Writer(char* COM_gtensor_file, long double *COM_gtensor_array, long *COM_gtensor_count_array):
     
    cdef:
        int i, j
        FILE* f_COM_gtensor
     
    f_COM_gtensor = fopen(COM_gtensor_file[:COM_gtensor_dist_file_size[0]], "a")
    for i in range(4 * box_size[0]):
        fprintf(f_COM_gtensor, "%d\t", i)
        for j in range(num_distinct_chains[0]):
            if COM_gtensor_count_array[i * num_distinct_chains[0] + j] == 0:
                fprintf(f_COM_gtensor, "%d\t%d\t", 0, 0)
            else:
                fprintf(f_COM_gtensor, "%LE\t%LE\t", (<long double> COM_gtensor_array[2 * i * num_distinct_chains[0] + 2 * j] / <long double> COM_gtensor_count_array[i * num_distinct_chains[0] + j]), (<long double> COM_gtensor_array[2 * i * num_distinct_chains[0] + 2 * j + 1] / <long double> COM_gtensor_count_array[i * num_distinct_chains[0] + j]))
        fprintf(f_COM_gtensor, "\n")
    fclose(f_COM_gtensor)

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cdef Nu_Writer(char* nu_file, long double *nu_clus_array,
               long double *nu_clus_err_array, long double *nu_solo_array,
               long double *nu_solo_err_array):
     
    cdef:
        int i
        FILE* f_nu
    f_nu = fopen(nu_file[:nu_file_size[0]], "a")
    for i in range(num_distinct_chains[0]):
        fprintf(f_nu, "%d\t%LE\t%LE\t%LE\t%LE\n", i, nu_clus_array[i], nu_clus_err_array[i],
                nu_solo_array[i], nu_solo_err_array[i])
    fclose(f_nu)

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cdef Rg_Full_Writer(char *rg_full_file, long double *rg_full_array,
                    long *rg_full_count_array):
     
    cdef:
        int i
        FILE* f_rg
    f_rg = fopen(rg_full_file[:rg_full_file_size[0]], "a")
    for i in range(num_distinct_chains[0]):
        fprintf(f_rg, "%d\t", i)
        if rg_full_count_array[i * RG_FULL_MAX + RG_FULL_DEN] > 0:
            fprintf(f_rg, "%LE\t", <long double> rg_full_array[i * RG_FULL_MAX + RG_FULL_DEN] / \
                    <long double> rg_full_count_array[i * RG_FULL_MAX + RG_FULL_DEN])
        else:
            fprintf(f_rg, "0\t")
        if rg_full_count_array[i * RG_FULL_MAX + RG_FULL_DIL] > 0:
            fprintf(f_rg, "%LE\n", <long double> rg_full_array[i * RG_FULL_MAX + RG_FULL_DIL] / \
                    <long double> rg_full_count_array[i * RG_FULL_MAX + RG_FULL_DIL])
        else:
            fprintf(f_rg, "0\n")
    fclose(f_rg)

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cdef COM_Bin_Count_Writer(char* COM_bin_count_file, long double *COM_bin_count_array, long *COM_count_array):
     
    cdef:
        int i, j
        FILE* f_COM_bin_count
     
    f_COM_bin_count = fopen(COM_bin_count_file[:COM_bin_count_dist_file_size[0]], "a")
    for i in range(4 * box_size[0]):
        fprintf(f_COM_bin_count, "%d\t", i)
        for j in range(num_distinct_chains[0]):
            if COM_count_array[i * num_distinct_chains[0] + j] > 0:
                COM_bin_count_array[i * num_distinct_chains[0] + j] /= COM_count_array[i * num_distinct_chains[0] + j]
            fprintf(f_COM_bin_count, "%LE\t", COM_bin_count_array[i * num_distinct_chains[0] + j])
        fprintf(f_COM_bin_count, "\n")
    fclose(f_COM_bin_count)

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cdef Rg_COM_Writer(char* rg_COM_file, long double *COM_Rg_array, long *COM_count_array):
     
    cdef:
        int i, j
        FILE* f_rg_COM
     
    f_rg_COM = fopen(rg_COM_file[:rg_COM_dist_file_size[0]], "a")
    for i in range(4 * box_size[0]):
        fprintf(f_rg_COM, "%d\t", i)
        for j in range(num_distinct_chains[0]):
            if COM_count_array[i * num_distinct_chains[0] + j] > 0:
                COM_Rg_array[i * num_distinct_chains[0] + j] /= COM_count_array[i * num_distinct_chains[0] + j]
            fprintf(f_rg_COM, "%LE\t", COM_Rg_array[i * num_distinct_chains[0] + j])
        fprintf(f_rg_COM, "\n")
    fclose(f_rg_COM)

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cdef Re_Dist_Writer_Clus(char* re_file_clus):
    cdef:
        int i
        FILE* f_re
         
    f_re = fopen(re_file_clus[:re_dist_file_size[0]], "a")
    for i in range(re_count_clus[0]):
        fprintf(f_re, "%d\t%LE\n", <int> re_dist_array_clus[2 * i], re_dist_array_clus[2 * i + 1])
        re_dist_array_clus[2 * i] = 0
        re_dist_array_clus[2 * i + 1] = 0
    re_count_clus[0] = 0
    fclose(f_re)

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cdef Re_Dist_Writer_Solo(char* re_file_solo):
    cdef:
        int i
        FILE* f_re
         
    f_re = fopen(re_file_solo[:re_dist_file_size[0]], "a")
    for i in range(re_count_solo[0]):
        fprintf(f_re, "%d\t%LE\n", <int> re_dist_array_solo[2 * i], re_dist_array_solo[2 * i + 1])
        re_dist_array_solo[2 * i] = 0
        re_dist_array_solo[2 * i + 1] = 0
    re_count_solo[0] = 0
    fclose(f_re)

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cdef Gtensor_Dist_Writer(char* gtensor_file):
    cdef:
        int i, j
        FILE* f_gtensor
         
    f_gtensor = fopen(gtensor_file[:gtensor_dist_file_size[0]], "a")
    for i in range(gtensor_count[0]):
        for j in range(6):
            fprintf(f_gtensor, "%LE\t", gtensor_dist_array[i * 6 + j])
            gtensor_dist_array[i * 6 + j] = 0
        fprintf(f_gtensor, "\n")
    gtensor_count[0] = 0
    fclose(f_gtensor)

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cdef Fractal_Ana_Dist_Writer(char* fractal_ana_file):
    cdef:
        int i, j
        FILE* f_fractal_ana
         
    f_fractal_ana = fopen(fractal_ana_file[:fractal_ana_dist_file_size[0]], "a")
    for i in range(fractal_ana_count[0]):
        for j in range(FRACTAL_MAX):
            fprintf(f_fractal_ana, "%LE\t", fractal_ana_dist_array[i * FRACTAL_MAX + j])
            fractal_ana_dist_array[i * FRACTAL_MAX + j] = 0
        fprintf(f_fractal_ana, "\n")
    fractal_ana_count[0] = 0
    fclose(f_fractal_ana)

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cdef RDF_Writer(char* RDF_file, long long *RDF_array):
     
    cdef:
        int i
        FILE* f_RDF
    f_RDF = fopen(RDF_file[:RDF_dist_file_size[0]], "a")
    for i in range(4 * box_size[0]):
        fprintf(f_RDF, "%d\t%lld\n", i, RDF_array[i])
    fclose(f_RDF)

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cdef mol_COM_Ana_Dist_Writer(long double *mol_COM_array, long double *mol_COM_array_old, char *mol_COM_file, char *mol_COM_file_2):
    cdef:
        int i, j
        long double cur_bin
        long double dist_sq
        long double cur_dist
        long double com_dist
        FILE* f_mol_COM
        FILE* f_mol_COM_2
        FILE* f_mol_COM_3
         
    f_mol_COM = fopen(mol_COM_file[:mol_COM_dist_file_size[0]], "a")
    for i in range(total_num_chains[0]):
        fprintf(f_mol_COM, "%d\t%LE\t%LE\t%LE\t%LE\n", i, mol_COM_array[4 * i + 3], mol_COM_array[4 * i], mol_COM_array[4 * i + 1], mol_COM_array[4 * i + 2])
    fclose(f_mol_COM)
     
    if mol_COM_array_old[0] + mol_COM_array_old[1] + mol_COM_array_old[2] + mol_COM_array_old[3] > 0:
        f_mol_COM_2 = fopen(mol_COM_file_2[:mol_COM_dist_file_size_2[0]], "a")
        for i in range(total_num_chains[0]):
            if mol_COM_array[i * 4 + 3] == 1 and mol_COM_array_old[i * 4 + 3] == 1:
                dist_sq = 0
                for j in range(3):
                    cur_dist = abs(mol_COM_array[i * 4 + j] - mol_COM_array_old[i * 4 + j])
                    if cur_dist > box_size[0] / 2:
                        cur_dist = box_size[0] - cur_dist
                    dist_sq += cur_dist ** 2
                fprintf(f_mol_COM_2, "%d\t%LE\n", i, dist_sq)
        fclose(f_mol_COM_2)
 
    for i in range(4 * total_num_chains[0]):
        mol_COM_array_old[i] = mol_COM_array[i]

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cdef mol_COM_Binned_Writer(char* mol_COM_file_3, long double *mol_COM_binned_array, long *mol_COM_binned_count_array):
     
    cdef:
        int i, j
        FILE* f_mol_COM_3
     
    f_mol_COM_3 = fopen(mol_COM_file_3[:mol_COM_dist_file_size_3[0]], "a")
    for i in range(4 * box_size[0]):
        fprintf(f_mol_COM_3, "%d\t", i)
        for j in range(num_distinct_chains[0]):
            if mol_COM_binned_count_array[i * num_distinct_chains[0] + j] == 0:
                fprintf(f_mol_COM_3, "%d\t", 0)
            else:
                fprintf(f_mol_COM_3, "%LE\t", i, (<long double> mol_COM_binned_array[i * num_distinct_chains[0] + j] / <long double> mol_COM_binned_count_array[i * num_distinct_chains[0] + j]))
        fprintf(f_mol_COM_3, "\n")
    fclose(f_mol_COM_3)

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cdef Sticker_Cluster_Writer(char* sticker_cluster_file, int *sticker_cluster_array):
     
    cdef:
        int i
        FILE* f_sticker_cluster
    f_sticker_cluster = fopen(sticker_cluster_file[:sticker_cluster_dist_file_size[0]], "a")
    for i in range(num_beads[0]):
        fprintf(f_sticker_cluster, "%d\t%LE\n", i, <long double> sticker_cluster_array[i] / <long double> total_num_cluster_stickers[0])
    fclose(f_sticker_cluster)

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cdef Crosslink_Ana_Writer(char* crosslink_file, long double *crosslink_ana_chain_array, long double *crosslink_ana_full_array, long *crosslink_ana_full_count_array):
    
    cdef:
        int i, j, k
        FILE* f_crosslink

    f_crosslink = fopen(crosslink_file[:crosslink_dist_file_size[0]], "a")
    for i in range(CROSSLINK_FULL_MAX):
        if i == 0:
            fprintf(f_crosslink, "Clus_All")
        elif i == 1:
            fprintf(f_crosslink, "Dil_All")
        elif i == 2:
            fprintf(f_crosslink, "Clus_Inter")
        else:
            fprintf(f_crosslink, "Dil_Inter")
        fprintf(f_crosslink, "\n")
        for j in range(bead_types[0]):
            for k in range(bead_types[0] + 1):
                if crosslink_ana_full_count_array[crosslink_full_conversion_array[i] * bead_types[0] + j] == 0:
                    fprintf(f_crosslink, "0\t")
                else:
                    fprintf(f_crosslink, "%LE\t", crosslink_ana_full_array[i * bead_types[0] * (bead_types[0] + 1) + j * (bead_types[0] + 1) + k] / <long double> crosslink_ana_full_count_array[crosslink_full_conversion_array[i] * bead_types[0] + j])
            fprintf(f_crosslink, "\n")
    fprintf(f_crosslink, "Clus_Inter_Normalized")
    fprintf(f_crosslink, "\n")
    for i in range(num_distinct_chains[0]):
        for j in range(num_distinct_chains[0]):
            fprintf(f_crosslink, "%LE\t", crosslink_ana_chain_array[i * num_distinct_chains[0] + j] / <long double> count_analysis[0])
    fprintf(f_crosslink, "\n")
    fclose(f_crosslink)

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cdef Cluster_Writer(char* cluster_file, int *cluster_size_dilute_array, int *cluster_size_dense_array):
    
    cdef:
        int i
        FILE* f_cluster
    f_cluster = fopen(cluster_file[:cluster_dist_file_size[0]], "a")
    for i in range(total_num_chains[0]):
        fprintf(f_cluster, "%d\t%d\t%d\n", i+1, cluster_size_dilute_array[i], cluster_size_dense_array[i])
    fclose(f_cluster)

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cdef Solvation_Writer(char* solvation_file):
     
    cdef:
        int i
        FILE* f_solvation
    f_solvation = fopen(solvation_file[:solvation_dist_file_size[0]], "a")
    for i in range(solvation_count[0]):
        fprintf(f_solvation, "%LE\t", solvation_array[i * SOLVATION_MAX + SOLVATION_CHAIN_TYPE])
        for j in range(SOLVATION_MAX - 1):
            fprintf(f_solvation, "%LE\t", solvation_array[i * SOLVATION_MAX + j])
            solvation_array[i * SOLVATION_MAX + j] = 0
        fprintf(f_solvation, "\n")
    fclose(f_solvation)
    solvation_count[0] = 0

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cdef Displacement_Writer(char* displacement_file):
     
    cdef:
        int i
        FILE* f_displacement
    f_displacement = fopen(displacement_file[:displacement_dist_file_size[0]], "a")
    for i in range(displacement_count[0]):
        fprintf(f_displacement, "%d\t%d\t%LE\n", <int> displacement_array[3 * i],
                <int> displacement_array[3 * i + 1], displacement_array[3 * i + 2])
        displacement_array[3 * i] = 0
        displacement_array[3 * i + 1] = 0
        displacement_array[3 * i + 2] = 0
    fclose(f_displacement)
    displacement_count[0] = 0

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cdef Connectedness_Writer(char *connectedness_file, long double *connectedness_array):

    cdef:
        int i, j, k, m
        FILE* f_connectedness
    f_connectedness = fopen(connectedness_file[:connectedness_dist_file_size[0]], "a")
    for i in range(connectedness_count[0]):
        for j in range(num_distinct_chains[0]):
            fprintf(f_connectedness, "#Chain Type %d\n", j)
            for k in range(connectedness_chains[0]):
                for m in range(CONNECTEDNESS_MAX):
                    fprintf(f_connectedness, "%LE\t", connectedness_array[i * num_distinct_chains[0] * connectedness_chains[0] * CONNECTEDNESS_MAX + j * connectedness_chains[0] * CONNECTEDNESS_MAX + k * CONNECTEDNESS_MAX + m])
                    connectedness_array[i * num_distinct_chains[0] * connectedness_chains[0] * CONNECTEDNESS_MAX + j * connectedness_chains[0] * CONNECTEDNESS_MAX + k * CONNECTEDNESS_MAX + m] = 0
                fprintf(f_connectedness, "\n")
        fprintf(f_connectedness, "#NEW_SET\n")
    fclose(f_connectedness)
    connectedness_count[0] = 0

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cdef Connectedness_Binned_Writer(char *connectedness_binned_file, long double *connectedness_binned_array,
                                 long *connectedness_binned_count_array):
    
    cdef:
        int i, j, k
        FILE* f_connectedness_binned
        
    f_connectedness_binned = fopen(connectedness_binned_file[:connectedness_binned_dist_file_size[0]], "a")
    for i in range(BETWEENNESS_BINS):
        fprintf(f_connectedness_binned, "%d\t", i)
        for j in range(num_distinct_chains[0]):
            if connectedness_binned_count_array[i * num_distinct_chains[0] + j] == 0:
                fprintf(f_connectedness_binned, "%d\t", 0)
            else:
                fprintf(f_connectedness_binned, "%LE\t", (<long double> connectedness_binned_array[i * num_distinct_chains[0] + j] / <long double> connectedness_binned_count_array[i * num_distinct_chains[0] + j]))
        fprintf(f_connectedness_binned, "\n")
    fclose(f_connectedness_binned)

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cdef Connectedness_Full_Writer(char *connectedness_full_file, long double *connectedness_full_array):

    cdef:
        int i, j, k
        FILE* f_connectedness_full
    f_connectedness_full = fopen(connectedness_full_file[:connectedness_full_dist_file_size[0]], "a")
    for i in range(connectedness_full_count[0]):
        fprintf(f_connectedness_full, "#NEW_FRAME\n")
        for j in range(total_num_chains[0]):
            for k in range(3):
                fprintf(f_connectedness_full, "%LE\t", connectedness_full_array[i * total_num_chains[0] * 3 + j * 3 + k])
            fprintf(f_connectedness_full, "\n")
    fclose(f_connectedness_full)
    connectedness_full_count[0] = 0

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cdef Sticker_Lifetime_Writer(char *sticker_lifetime_file, int *sticker_lifetime_array, int final_check):

    cdef:
        int i
        FILE* f_sticker_lifetime
    f_sticker_lifetime = fopen(sticker_lifetime_file[:sticker_lifetime_dist_file_size[0]], "a")
    for i in range(sticker_lifetime_count[0]):
        fprintf(f_sticker_lifetime, "%d\n", sticker_lifetime_dist_array[i])
        sticker_lifetime_dist_array[i] = 0
    sticker_lifetime_count[0] = 0
    
    if final_check == 1:
        fprintf(f_sticker_lifetime, "#Remainder\n")
        for i in range(<int> ((total_num_stickers[0] * total_num_stickers[0] - total_num_stickers[0]) / 2)):
            if sticker_lifetime_array[i] > 0:
                fprintf(f_sticker_lifetime, "%d\n", sticker_lifetime_array[i])
        
    fclose(f_sticker_lifetime)

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cdef Concave_Hull_Writer(char *concave_hull_file, int *concave_hull_array, long double *concave_hull_com_array):

    cdef:
        int i, j, k
        FILE* f_concave_hull
    f_concave_hull = fopen(concave_hull_file[:concave_hull_dist_file_size[0]], "a")
    for i in range(count_analysis[0]):
        fprintf(f_concave_hull, "#NewFrame\n")
        fprintf(f_concave_hull, "%.2Lf,%.2Lf\n", concave_hull_com_array[i * 2], concave_hull_com_array[i * 2 + 1])
        for j in range(box_size[0]):
            for k in range(box_size[0]):
                if concave_hull_array[i * box_size[0] * box_size[0] + j * box_size[0] + k] == 1:
                    fprintf(f_concave_hull, "%d,%d,", j, k)
        fprintf(f_concave_hull, "\n")
            
    fclose(f_concave_hull)

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cdef Nu_Calculator(const long double *scaling_array, const int chain_type,
                   const int *scaling_length_fb_array, const int *scaling_length_array,
                   const int *chain_length_array, const int end_cutoff,
                   const int length_cutoff): 
    cdef:
        int i, j
        int fb, lb
        int chain_length, scaling_length
        int nu_counter = 0
        double r1, r2, r3, r4, dr1, dr2, d1, d2
        double *nu_array
    
    chain_length = chain_length_array[chain_type]
    scaling_length = scaling_length_array[chain_type]
    first_scaling_site = scaling_length_fb_array[chain_type]
    last_scaling_site = first_scaling_site + scaling_length
    nu_array = <double *> malloc(int((scaling_length ** 2) / 2) * sizeof(double))
    nu_mean[0] = 0
    nu_err[0] = 0
     
    r1 = 0
    r2 = 0
    for i in range(first_scaling_site, last_scaling_site - 1):
        # r1 and r2 are two residues that are more than 'chain_length' apart and 'end_cutoff' away from the end of the polymer
        # dr1 is the distance along the chain between r1 and r2
        # d1 is the  distance in space between r1 and r2
        r2 += 1
        if r2 == chain_length:
            r1 += 1
            r2 = r1 + 1
        dr1 = abs(r1 - r2)
        if r1 < end_cutoff or r1 > (chain_length - end_cutoff - 1) or r2 > (chain_length - end_cutoff - 1) or dr1 < length_cutoff:
            continue
        d1 = sqrt(scaling_array[i])
 
        r3 = r1
        r4 = r2      
        for j in range(i + 1, last_scaling_site):
            # r3 and r4 are two residues that are more than 'chain_length' apart and 'end_cutoff' away from the end of the polymer
            # dr2 is the distance along the chain between r3 and r4
            # d2 is the  distance in space between r3 and r4
            r4 += 1
            if r4 == chain_length:
                r3 += 1
                r4 = r3 + 1
            dr2 = abs(r3 - r4)
            if r3 < end_cutoff or r3 > (chain_length - end_cutoff - 1) or r4 > (chain_length - end_cutoff - 1) or dr2 < length_cutoff or dr1 == dr2:
                continue
            d2 = sqrt(scaling_array[j])
             
            # our_nu is the sum of the pairwise scaling exponents, to be averaged at the end
            # If one wants a histogram and not just the average, this should instead be a list or array of pairwise scaling exponents
            nu_mean[0] += (log(d1) - log(d2)) / (log(dr1) - log(dr2))
            nu_array[nu_counter] = (log(d1) - log(d2)) / (log(dr1) - log(dr2))
            nu_counter += 1
     
    # We divide here to average the scaling exponents and return the final scaling exponent
    if nu_counter > 0:
        
        nu_mean[0] /= nu_counter
     
        for i in range(nu_counter):
            nu_err[0] += (nu_array[i] - nu_mean[0]) ** 2
         
        nu_err[0] = sqrt(nu_err[0] / (nu_counter - 1))
        nu_err[0] /= sqrt(nu_counter)
         
    free(nu_array)

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cdef Bin_Count_Calculator(int *frame_array, int fb, int lb, int *system_COM):
    cdef:
        int i, bin_count
        int *bin_count_array
         
    bin_count_array = <int *> calloc(4 * box_size[0], sizeof(int))
     
    for i in range(fb, lb):
        our_bin = <int> (4 * sqrt(Dist_Bead_Point_Squared(frame_array, i, system_COM)))
        if bin_count_array[our_bin] == 0:
            bin_count_array[our_bin] = 1
     
    for i in range(4 * box_size[0]):
        bin_count += bin_count_array[i]
 
    free(bin_count_array)
     
    return bin_count
         
@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cdef Rg_Calculator(int *frame_array, int clus_check, int first_bead, int chain_length):
    cdef:
        int i, j
        long double rg
     
    rg = 0
    if clus_check == 1:
        for i in range(chain_length - 1):
            for j in range(i + 1, chain_length):
                rg += Dist_Bead_Bead_Squared_No_PBC(frame_array, first_bead + i, first_bead + j)
    else:
        for i in range(chain_length - 1):
            for j in range(i + 1, chain_length):
                rg += Dist_Bead_Bead_Squared(frame_array, first_bead + i, first_bead + j)        
    return rg / (chain_length * chain_length)

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cdef Chain_Asph_Acyl_Calculator(int *frame_array, long double *asph_acyl_array,
                                const int *chain_id_to_distinct_chain_array,
                                const int *chain_id_to_first_bead_array,
                                const int *chain_length_array):
    cdef:
        int chain, i, j
        int chain_length, fb, lb
        long double asph
        long double acyl
        long double gtensor[6]
        long double eg_array[3]
     
    for chain in range(total_num_chains[0]):
        chain_length = chain_length_array[chain_id_to_distinct_chain_array[chain]]
        fb = chain_id_to_first_bead_array[chain]
        lb = fb + chain_length
        for i in range(6):
            gtensor[i] = 0
     
        for i in range(3):
            eg_array[i] = 0
             
        for i in range(fb, lb - 1):
            for j in range(i + 1, lb):
                for k in range(3):
                    gtensor[k] += Dist_Bead_Bead_1D(frame_array, i, j, k) * Dist_Bead_Bead_1D(frame_array, i, j, k)
                    gtensor[k + 3] += Dist_Bead_Bead_1D(frame_array, i, j, k) * Dist_Bead_Bead_1D(frame_array, i, j, (k + 1) % 3)
         
        for i in range(6):
            gtensor[i] /= (chain_length * chain_length)
         
        eigvals = np.linalg.eigvals(np.array([[gtensor[0], gtensor[3], gtensor[5]],
                                              [gtensor[3], gtensor[1], gtensor[4]],
                                              [gtensor[5], gtensor[4], gtensor[2]]]))
         
        for i in range(3):
            eg_array[i] = <long double> eigvals[i]
         
        asph = 1 - 3 * ((eg_array[0] * eg_array[1] + eg_array[1] * eg_array[2] + eg_array[2] * eg_array[0]) / ((eg_array[0] + eg_array[1] + eg_array[2]) ** 2))
         
        if eg_array[0] <= eg_array[1]:
            if eg_array[1] <= eg_array[2]:
                acyl = abs(eg_array[1] - eg_array[0])
            else:
                acyl = abs(eg_array[2] - eg_array[0])
        else:
            if eg_array[2] <= eg_array[0]:
                acyl = abs(eg_array[1] - eg_array[2])
            else:
                acyl = abs(eg_array[1] - eg_array[0])
         
        asph_acyl_array[2 * chain] = asph
        asph_acyl_array[2 * chain + 1] = acyl
         
    return

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cdef Partial_Rg_Calculator(int *frame_array, int clus_check, int b_initial, int b_final):
    cdef:
        int i, j
        long double rg
     
    rg = 0
    if clus_check == 1:
        for i in range(b_initial, b_final - 1):
            for j in range(i + 1, b_final):
                rg += Dist_Bead_Bead_Squared_No_PBC(frame_array, i, j)
    else:
        for i in range(b_initial, b_final - 1):
            for j in range(i + 1, b_final):
                rg += Dist_Bead_Bead_Squared(frame_array, i, j)        
    return rg / ((b_final - b_initial) * (b_final - b_initial))

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.        
cdef Betweenness_Calculator(int *degree_checklist,
                            long double *connectedness_betweenness_array):

    cdef:
        int i
    
    betweenness_array = np.zeros([total_num_chains[0], total_num_chains[0]], dtype = np.int32)
    for i in range(total_num_chains[0]):
        for j in range(total_num_chains[0]):
            betweenness_array[i][j] = degree_checklist[i * total_num_chains[0] + j]
    betweenness_array_new = betweenness_calculator.Betweenness(betweenness_array)
    for i in range(total_num_chains[0]):
        connectedness_betweenness_array[i] = betweenness_array_new[i]

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cdef COM_Distance_Analysis(int *frame_array, int *topo_array, int *lattice_array, int *system_COM,
                           long double *COM_Rg_array, long double *COM_bin_count_array,
                           long *COM_bead_array, long *COM_chain_array, long *COM_count_array,
                           long *COM_sticker_array, long *COM_sticker_count_array,
                           long double *COM_Re_array, long *COM_Re_count_array,
                           long double *COM_angle_array, long *COM_angle_count_array,
                           long double *COM_dot_prod_array, long *COM_dot_prod_count_array,
                           long double *COM_solvation_array, long *COM_solvation_count_array,
                           long double *rg_array, int *bin_count_array,
                           long double *cosine_array, int *sticker_array,
                           const int *chain_id_to_distinct_chain_array,
                           const int *chain_id_to_first_bead_array,
                           const int *chain_length_array, const int *chain_length_fb_array):
 
    cdef:
        int i, j, k, m, n
        int b_initial, b_final, cur_dist
        int bin_1, bin_2, chain_1, chain_2
        int chain_1_type, chain_1_length
        int bead_type_1, bead_type_2
        int sticker_check, inter_sticker_check
        int empty_site, chain_degrees
        int chain_length_fb
        int tmpR1[3]
        int tmpR2[3]
        int *chain_degree_checklist
        int *bin_beads_array
        int *bin_chains_array
        long *COM_chain_array_temp
        long double length_1, length_2, dot_prod
     
    COM_chain_array_temp = <long *> calloc(4 * box_size[0] * total_num_chains[0], sizeof(long))
    chain_degree_checklist = <int *> malloc(total_num_chains[0] * sizeof(int))
    bin_beads_array = <int *> malloc(num_distinct_chains[0] * sizeof(int))
    bin_chains_array = <int *> malloc(num_distinct_chains[0] * sizeof(int))
     
    for i in range(num_beads[0]):
        sticker_check = 0
        inter_sticker_check = 0
        bin_1 = <int> (4 * sqrt(Dist_Bead_Point_Squared(frame_array, i, system_COM)))
        chain_1 = topo_array[TOPO_ARRAY_MAX * i + TOPO_ARRAY_CHAIN]
        bead_type_1 = topo_array[TOPO_ARRAY_MAX * i + TOPO_ARRAY_BEAD]
        chain_1_type = chain_id_to_distinct_chain_array[chain_1]
        #chain_1_length = chain_length_array[chain_1_type]
        chain_1_length = 101
        
        COM_Rg_array[num_distinct_chains[0] * bin_1 + chain_1_type] += rg_array[chain_1]
        COM_bin_count_array[num_distinct_chains[0] * bin_1 + chain_1_type] += bin_count_array[chain_1]
        COM_bead_array[bead_types[0] * bin_1 + bead_type_1] += 1
        COM_chain_array_temp[total_num_chains[0] * bin_1 + chain_1] += 1
        COM_count_array[num_distinct_chains[0] * bin_1 + chain_1_type] += 1
        if i < num_beads[0] - 5:
            if chain_1 == topo_array[TOPO_ARRAY_MAX * (i + 5) + TOPO_ARRAY_CHAIN]:
                COM_Re_array[num_distinct_chains[0] * bin_1 + chain_1_type] += Dist_Bead_Bead_Squared(frame_array, i, i+5)
                COM_Re_count_array[num_distinct_chains[0] * bin_1 + chain_1_type] += 1
        if i < num_beads[0] - chain_1_length - 1:
            if chain_1 == topo_array[TOPO_ARRAY_MAX * (i + chain_1_length - 1) + TOPO_ARRAY_CHAIN]:
                if cosine_array[2 * i] <= 1:
                    COM_angle_array[num_distinct_chains[0] * bin_1 + chain_1_type] += cosine_array[2 * i]
                    COM_angle_count_array[num_distinct_chains[0] * bin_1 + chain_1_type] += 1
                if cosine_array[2 * (i + chain_1_length - 1) + 1] <= 1:
                    bin_2 = <int> (4 * sqrt(Dist_Bead_Point_Squared(frame_array, i + chain_1_length - 1, system_COM)))
                    COM_angle_array[num_distinct_chains[0] * bin_2 + chain_1_type] += cosine_array[2 * (i + chain_1_length - 1) + 1]
                    COM_angle_count_array[num_distinct_chains[0] * bin_2 + chain_1_type] += 1
        for m in range(num_stickers[0]):
            if bead_type_1 == sticker_array[m]:
                for j in range(3):
                    tmpR1[j] = frame_array[FRAME_ARRAY_MAX * i + j]
                for j in range(27):
                    if j == 13:
                        continue
                    new_bead = Rotation_Check(lattice_array, tmpR1, j)
                    #We found a bead
                    if new_bead != -1:
                        chain_2 = topo_array[TOPO_ARRAY_MAX * new_bead + TOPO_ARRAY_CHAIN]
                        bead_type_2 = topo_array[TOPO_ARRAY_MAX * new_bead + TOPO_ARRAY_BEAD]
                        for n in range(num_stickers[0]):
                            if bead_type_2 == sticker_array[n]:
                                if sticker_check == 0:
                                    COM_sticker_array[4 * bin_1] += 1
                                    sticker_check = 1
                                COM_sticker_array[4 * bin_1 + 1] += 1
                                if chain_2 != chain_1:
                                    if inter_sticker_check == 0:
                                        COM_sticker_array[4 * bin_1 + 2] += 1
                                        inter_sticker_check = 1
                                    COM_sticker_array[4 * bin_1 + 3] += 1  
                                break 
                COM_sticker_count_array[bin_1] += 1
                break
      
    for i in range(4 * box_size[0]):
        for j in range(num_distinct_chains[0]):
            bin_beads_array[j] = 0
            bin_chains_array[j] = 0
        for j in range(total_num_chains[0]):
            chain_1_type = chain_id_to_distinct_chain_array[j]
            if COM_chain_array_temp[i * total_num_chains[0] + j] > 0:
                bin_beads_array[chain_1_type] += COM_chain_array_temp[i * total_num_chains[0] + j]
                bin_chains_array[chain_1_type] += 1
        for j in range(num_distinct_chains[0]):
            COM_chain_array[2 * (num_distinct_chains[0] * i + j)] += bin_chains_array[j]
            COM_chain_array[2 * (num_distinct_chains[0] * i + j) + 1] += bin_beads_array[j]
      
    COM_chain_count[0] += 1
      
    #===========================================================================
    # for i in range(total_num_chains[0]):
    #     empty_site = 0
    #     cur_chain_site = 0
    #     other_chain_site = 0
    #     chain_degrees = 0
    #     chain_1_type = chain_id_to_distinct_chain_array[i]
    #     b_initial = chain_id_to_first_bead_array[i]
    #     b_final = b_initial + chain_length_array[chain_1_type]
    #     for j in range(total_num_chains[0]):
    #         chain_degree_checklist[j] = 0
    #     for j in range(b_initial, b_final):
    #         for m in range(3):
    #             tmpR1[m] = frame_array[FRAME_ARRAY_MAX * j + m]
    #         for m in range(27):
    #             if m == 13:
    #                 continue
    #             neighbor = Rotation_Check(lattice_array, tmpR1, m)
    #             if neighbor == -1:
    #                 empty_site += 1
    #             elif topo_array[TOPO_ARRAY_MAX * neighbor + TOPO_ARRAY_CHAIN] == i:
    #                 cur_chain_site += 1
    #             else:
    #                 other_chain_site += 1
    #                 chain_degree_checklist[topo_array[TOPO_ARRAY_MAX * neighbor + TOPO_ARRAY_CHAIN]] = 1
    #     for j in range(total_num_chains[0]):
    #         if chain_degree_checklist[j] == 1:
    #             chain_degrees += 1
    #     for j in range(b_initial, b_final):
    #         bin_1 = <int> (4 * sqrt(Dist_Bead_Point_Squared(frame_array, j, system_COM)))
    #         COM_solvation_array[bin_1 * num_distinct_chains[0] * COM_SOLVATION_MAX + chain_1_type * COM_SOLVATION_MAX + COM_SOLVATION_EMPTY] += empty_site
    #         COM_solvation_array[bin_1 * num_distinct_chains[0] * COM_SOLVATION_MAX + chain_1_type * COM_SOLVATION_MAX + COM_SOLVATION_SELF] += cur_chain_site
    #         COM_solvation_array[bin_1 * num_distinct_chains[0] * COM_SOLVATION_MAX + chain_1_type * COM_SOLVATION_MAX + COM_SOLVATION_OTHER] += other_chain_site
    #         COM_solvation_array[bin_1 * num_distinct_chains[0] * COM_SOLVATION_MAX + chain_1_type * COM_SOLVATION_MAX + COM_SOLVATION_DEGREES] += chain_degrees
    #         COM_solvation_count_array[bin_1 * num_distinct_chains[0] + chain_1_type] += 1        
    #===========================================================================
  
    #===========================================================================
    # for i in range(total_num_chains[0]):
    #     chain_1_type = chain_id_to_distinct_chain_array[i]
    #     b_initial = chain_id_to_first_bead_array[i]
    #     b_final = b_initial + chain_length_array[chain_1_type]
    #     chain_length_fb = chain_length_fb_array[chain_1_type]
    #     for j in range(b_initial, b_final - 2):
    #         length_1 = 0
    #         for m in range(3):
    #             cur_dist = frame_array[FRAME_ARRAY_MAX * (j + 1) + m] - frame_array[FRAME_ARRAY_MAX * j + m]
    #             if cur_dist > box_size[0] / 2:
    #                 cur_dist = cur_dist - box_size[0]
    #             elif cur_dist < - box_size[0] / 2:
    #                 cur_dist = cur_dist + box_size[0]
    #             tmpR1[m] = cur_dist
    #             length_1 += cur_dist * cur_dist
    #         for k in range(j + 1, b_final - 1):
    #             length_2 = 0
    #             dot_prod = 0
    #             for m in range(3):
    #                 cur_dist = frame_array[FRAME_ARRAY_MAX * (k + 1) + m] - frame_array[FRAME_ARRAY_MAX * k + m]
    #                 if cur_dist > box_size[0] / 2:
    #                     cur_dist = cur_dist - box_size[0]
    #                 elif cur_dist < - box_size[0] / 2:
    #                     cur_dist = cur_dist + box_size[0]
    #                 tmpR2[m] = cur_dist
    #                 length_2 += cur_dist * cur_dist
    #                 dot_prod += tmpR1[m] * tmpR2[m]
    #             dot_prod /= (sqrt(length_1) * sqrt(length_2))
    #             cur_dist = k - j
    #             for m in range(j, k + 2):
    #                 bin_1 = <int> (4 * sqrt(Dist_Bead_Point_Squared(frame_array, m, system_COM)))
    #                 COM_dot_prod_array[(chain_length_fb + cur_dist) * 4 * box_size[0] + bin_1] += dot_prod
    #                 COM_dot_prod_count_array[(chain_length_fb + cur_dist) * 4 * box_size[0] + bin_1] += 1 
    #===========================================================================
         
    free(COM_chain_array_temp)
    free(chain_degree_checklist)
    free(bin_beads_array)
    free(bin_chains_array)

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cdef Solvation_Analysis(int *frame_array, int *topo_array, int *lattice_array, int *cluster_list_largest,
                        const int clus_size_largest, long double *rg_array,
                        long double *partial_rg_array,
                        const int *chain_id_to_distinct_chain_array,
                        const int *chain_id_to_first_bead_array,
                        const int *chain_length_array, char *solvation_file):
 
    cdef:
        int i, j, m
        int cur_chain, chain_type, chain_length
        int b_initial, b_final, neighbor
        int partial_section, partial_solvation_site, partial_self_site, partial_chain_site
        int tmpR1[3]
        int *chain_degree_checklist
     
    chain_degree_checklist = <int *> malloc(total_num_chains[0] * (PARTIAL_RG_NUM + 1) * sizeof(int))
     
    for i in range(clus_size_largest):
        cur_chain = cluster_list_largest[i]
        chain_type = chain_id_to_distinct_chain_array[cur_chain]
        chain_length = chain_length_array[chain_type]
        b_initial = chain_id_to_first_bead_array[cur_chain]
        b_final = b_initial + chain_length
        for j in range((PARTIAL_RG_NUM + 1) * total_num_chains[0]):
            chain_degree_checklist[j] = 0
        for j in range(b_initial, b_final):
            partial_section = <int> (PARTIAL_RG_NUM * (j - b_initial) / chain_length)
            partial_solvation_site = SOLVATION_MAX_NUM * (partial_section + 1) + SOLVATION_EMPTY_SITE
            partial_self_site = SOLVATION_MAX_NUM * (partial_section + 1) + SOLVATION_SELF_SITE
            partial_chain_site = SOLVATION_MAX_NUM * (partial_section + 1) + SOLVATION_CHAIN_SITE
            for m in range(3):
                tmpR1[m] = frame_array[FRAME_ARRAY_MAX * j + m]
            for m in range(27):
                if m == 13:
                    continue
                neighbor = Rotation_Check(lattice_array, tmpR1, m)
                if neighbor == -1:
                    solvation_array[solvation_count[0] * SOLVATION_MAX + SOLVATION_EMPTY_SITE] += 1
                    solvation_array[solvation_count[0] * SOLVATION_MAX + partial_solvation_site] += 1
                elif topo_array[TOPO_ARRAY_MAX * neighbor + TOPO_ARRAY_CHAIN] == cur_chain:
                    solvation_array[solvation_count[0] * SOLVATION_MAX + SOLVATION_SELF_SITE] += 1
                    solvation_array[solvation_count[0] * SOLVATION_MAX + partial_self_site] += 1
                else:
                    solvation_array[solvation_count[0] * SOLVATION_MAX + SOLVATION_CHAIN_SITE] += 1
                    solvation_array[solvation_count[0] * SOLVATION_MAX + partial_chain_site] += 1
                    chain_degree_checklist[topo_array[TOPO_ARRAY_MAX * neighbor + TOPO_ARRAY_CHAIN]] = 1
                    chain_degree_checklist[(partial_section + 1) * total_num_chains[0] + topo_array[TOPO_ARRAY_MAX * neighbor + TOPO_ARRAY_CHAIN]] = 1
                    
        solvation_array[solvation_count[0] * SOLVATION_MAX + SOLVATION_RG] += rg_array[cur_chain]
        for j in range(PARTIAL_RG_NUM):
            solvation_array[solvation_count[0] * SOLVATION_MAX + SOLVATION_MAX_NUM * (j + 1) + SOLVATION_RG] += partial_rg_array[cur_chain * PARTIAL_RG_NUM + j]
        for j in range(total_num_chains[0]):
            if chain_degree_checklist[j] == 1:
                solvation_array[solvation_count[0] * SOLVATION_MAX + SOLVATION_DEGREES] += 1
                for m in range(PARTIAL_RG_NUM):
                    if chain_degree_checklist[(m + 1) * total_num_chains[0] + j] == 1:
                        solvation_array[solvation_count[0] * SOLVATION_MAX + SOLVATION_MAX_NUM * (m + 1) + SOLVATION_DEGREES] += 1
        
        solvation_array[solvation_count[0] * SOLVATION_MAX + SOLVATION_CHAIN_TYPE] = chain_type
        solvation_count[0] += 1
        if solvation_count[0] == SOLVATION_ANA_DIST_ARRAY_SIZE:
            Solvation_Writer(solvation_file)
     
    free(chain_degree_checklist)
         
@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cdef COM_GTensor_Analysis(int *frame_array, int *topo_array, long double *COM_gtensor_array,
                          long *COM_gtensor_count_array, long double *asph_acyl_array,
                          int *system_COM, const int *chain_id_to_distinct_chain_array):
     
    cdef:
        int i, our_bin, our_chain, chain_type
         
    for i in range(num_beads[0]):
        our_bin = <int> (4 * sqrt(Dist_Bead_Point_Squared(frame_array, i, system_COM)))
        our_chain = topo_array[TOPO_ARRAY_MAX * i + TOPO_ARRAY_CHAIN]
        chain_type = chain_id_to_distinct_chain_array[our_chain]
        COM_gtensor_array[2 * num_distinct_chains[0] * our_bin + 2 * chain_type] += asph_acyl_array[2 * our_chain]
        COM_gtensor_array[2 * num_distinct_chains[0] * our_bin + 2 * chain_type + 1] += asph_acyl_array[2 * our_chain + 1]
        COM_gtensor_count_array[num_distinct_chains[0] * our_bin + chain_type] += 1

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cdef Bin_Count_Analysis(int *frame_array, int *bin_count_array, int *system_COM,
                        int *chain_id_to_distinct_chain_array,
                        int *chain_id_to_first_bead_array, int *chain_length_array):
     
    cdef:
        int i, fb, lb
    
    for i in range(total_num_chains[0]):
        fb = chain_id_to_first_bead_array[i]
        lb = fb + chain_length_array[chain_id_to_distinct_chain_array[i]]
        bin_count_array[i] = Bin_Count_Calculator(frame_array, fb, lb, system_COM)
 
@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cdef Rg_Analysis(int *frame_array, long double *rg_array, int *cluster_list_largest,
                 const int clus_size, int *chain_id_to_distinct_chain_array, 
                 int *chain_id_to_first_bead_array, int *chain_length_array):
     
    cdef:
        int i
        int *clus_checklist
 
    clus_checklist = <int *> calloc(total_num_chains[0], sizeof(int))
     
    for i in range(clus_size):
        clus_checklist[cluster_list_largest[i]] = 1
     
    for i in range(total_num_chains[0]):
        if clus_checklist[i] == 1:
            rg_array[i] = Rg_Calculator(frame_array, 1, chain_id_to_first_bead_array[i], chain_length_array[chain_id_to_distinct_chain_array[i]])
        else:
            rg_array[i] = Rg_Calculator(frame_array, 0, chain_id_to_first_bead_array[i], chain_length_array[chain_id_to_distinct_chain_array[i]])
    
    free(clus_checklist)
    
@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cdef Partial_Rg_Analysis(int *frame_array, long double *partial_rg_array,
                         int *cluster_list_largest, const int clus_size,
                         const int *chain_id_to_distinct_chain_array,
                         const int *chain_id_to_first_bead_array,
                         const int *chain_length_array):
     
    cdef:
        int i, j
        int b_initial, b_final
        int chain_type, chain_length
        int *clus_checklist
 
    clus_checklist = <int *> calloc(total_num_chains[0], sizeof(int))
     
    for i in range(clus_size):
        clus_checklist[cluster_list_largest[i]] = 1
     
    for i in range(total_num_chains[0]):
        chain_type = chain_id_to_distinct_chain_array[i]
        chain_length = chain_length_array[chain_type]
        b_initial = chain_id_to_first_bead_array[i]
        for j in range(PARTIAL_RG_NUM):
            b_final = b_initial + <int> (chain_length * (j + 1) / PARTIAL_RG_NUM)
            b_initial += <int> (chain_length * j / PARTIAL_RG_NUM)
            if clus_checklist[i] == 1:
                partial_rg_array[i * PARTIAL_RG_NUM + j] = Partial_Rg_Calculator(frame_array, 1, b_initial, b_final)
            else:
                partial_rg_array[i * PARTIAL_RG_NUM + j] = Partial_Rg_Calculator(frame_array, 0, b_initial, b_final)
    
    free(clus_checklist)
    
@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cdef COM_Analysis(int *frame_array, int *system_COM, long double *cluster_COM,
                  int *cluster_list_largest, int clus_size_largest,
                  int *chain_id_to_distinct_chain_array, int *chain_id_to_first_bead_array,
                  int *chain_length_array):

    cdef:
        int i, j, k
        int fb, lb
        int clus_beads = 0
        int *clus_checklist
    
    clus_checklist = <int *> calloc(total_num_chains[0], sizeof(int))

    for i in range(3):
        system_COM[i] = 0
        cluster_COM[i] = 0
    
    for i in range(clus_size_largest):
        clus_checklist[cluster_list_largest[i]] = 1
    
    for i in range(total_num_chains[0]):
        
        fb = chain_id_to_first_bead_array[i]
        lb = fb + chain_length_array[chain_id_to_distinct_chain_array[i]]
        
        if clus_checklist[i] == 1:
            for j in range(fb, lb):
                clus_beads += 1
                for k in range(3):
                    system_COM[k] += frame_array[FRAME_ARRAY_MAX * j + k]
                    cluster_COM[k] += frame_array[FRAME_ARRAY_MAX * j + k]
        else:
            for j in range(fb, lb):
                for k in range(3):
                    system_COM[k] += frame_array[FRAME_ARRAY_MAX * j + k]
    
    for i in range(3):
        system_COM[i] = <int> (system_COM[i] / num_beads[0])
        cluster_COM[i] = cluster_COM[i] / clus_beads
    
    free(clus_checklist)
 
@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cdef Cosine_Analysis(int *frame_array, int *topo_array,
                     long double *cosine_array, int *system_COM,
                     int *chain_id_to_distinct_chain_array, int *chain_length_array):
     
    cdef:
        int i
        int chain_1, chain_2
        int chain_length
     
    for i in range(num_beads[0]):
        chain_1 = topo_array[TOPO_ARRAY_MAX * i + TOPO_ARRAY_CHAIN]
        #chain_length = chain_length_array[chain_id_to_distinct_chain_array[chain_1]]
        chain_length = 100
        if i + chain_length - 1 >= num_beads[0]:
            cosine_array[2 * i] = 5
            continue
        if i < chain_length - 1:
            cosine_array[2 * i + 1] = 5
        chain_1 = topo_array[TOPO_ARRAY_MAX * i + TOPO_ARRAY_CHAIN]
        chain_2 = topo_array[TOPO_ARRAY_MAX * (i + chain_length - 1) + TOPO_ARRAY_CHAIN]
        if chain_1 == chain_2:
            cosine_array[2 * i] = Cos_Angle_Two_Beads_Point(frame_array, i, i + chain_length - 1, system_COM)
            cosine_array[2 * (i + chain_length - 1) + 1] = Cos_Angle_Two_Beads_Point(frame_array, i + chain_length - 1, i, system_COM)
            #We want the square of the cosine
            cosine_array[2 * i] *= cosine_array[2 * i]
            cosine_array[2 * (i + chain_length - 1) + 1] *= cosine_array[2 * (i + chain_length - 1) + 1]
        else:
            cosine_array[2 * i] = 5
            cosine_array[2 * (i + chain_length - 1) + 1] = 5
    
@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cdef RDF_Analysis(int *frame_array, int *topo_array, long long *RDF_array, int *sticker_array,
                  const int *cluster_list_largest, const int clus_size,
                  const int *chain_id_to_distinct_chain_array,
                  const int *chain_id_to_first_bead_array, int *chain_length_array):
     
    cdef:
        int i, j, k, c1, b1, b2
        int type_b1, type_b2, cur_bin
        int num_clus_beads = 0
        int bead_counter = 0
        int *clus_beads_array

    for i in range(clus_size):
        num_clus_beads += chain_length_array[chain_id_to_distinct_chain_array[cluster_list_largest[i]]]
    
    clus_beads_array = <int *> malloc(num_clus_beads * sizeof(int))
    for i in range(clus_size):
        c1 = cluster_list_largest[i]
        b1 = chain_id_to_first_bead_array[c1]
        b2 = b1 + chain_length_array[chain_id_to_distinct_chain_array[c1]]
        for j in range(b1, b2):
            clus_beads_array[bead_counter] = j
            bead_counter += 1
    
    for i in range(num_clus_beads - 1):
        b1 = clus_beads_array[i]
        type_b1 = topo_array[TOPO_ARRAY_MAX * b1 + TOPO_ARRAY_BEAD]
        for j in range(num_stickers[0]):
            if type_b1 == sticker_array[j]:
                break
        else:
            continue
        for j in range(i + 1, num_clus_beads):
            b2 = clus_beads_array[j]
            type_b2 = topo_array[TOPO_ARRAY_MAX * b2 + TOPO_ARRAY_BEAD]
            for k in range(num_stickers[0]):
                if type_b2 == sticker_array[k]:
                    break
            else:
                continue
            cur_bin = <int> (4 * sqrt(Dist_Bead_Bead_Squared_No_PBC(frame_array, b1, b2)))
            RDF_array[cur_bin] += 1
    
    free(clus_beads_array)

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cdef Cluster_Analysis(int *frame_array, int *topo_array, int *lattice_array, int *cluster_list_largest,
                      int *cluster_size_dilute_array, int *cluster_size_dense_array,
                      int *chain_id_to_distinct_chain_array,
                      int *chain_id_to_first_bead_array, int *chain_length_array):

    cdef:
        int i, j, k, m
        int cur_chain
        int cur_largest = 0
        int clus_size
        int fb, lb
        int new_bead
        int new_chain
        int tmpR1[3]
        int *cluster_list
        int *cur_clust_checklist
        int *chain_checklist
        int *cluster_size_list

    cluster_list = <int *> malloc(total_num_chains[0] * sizeof(int))
    cur_clust_checklist = <int *> malloc(total_num_chains[0] * sizeof(int))
    chain_checklist = <int *> malloc(total_num_chains[0] * sizeof(int))
    cluster_size_list = <int *> malloc(total_num_chains[0] * sizeof(int))

    for i in range(total_num_chains[0]):
        chain_checklist[i] = 0
        cluster_size_list[i] = 0

    for i in range(total_num_chains[0]):
        if chain_checklist[i] == 1:
            continue
        for j in range(total_num_chains[0]):
            cluster_list[j] = -1
            cur_clust_checklist[j] = 0

        cur_clust_checklist[i] = 1
        cluster_list[0] = i
        clus_size = 1

        for j in range(total_num_chains[0]):
            cur_chain = cluster_list[j]
            if cur_chain == -1:
                break
            fb = chain_id_to_first_bead_array[cur_chain]
            lb = fb + chain_length_array[chain_id_to_distinct_chain_array[cur_chain]]
            for k in range(fb, lb):
                for m in range(3):
                    tmpR1[m] = frame_array[FRAME_ARRAY_MAX * k + m]
                for m in range(27):
                    if m == 13:
                        continue
                    new_bead = Rotation_Check(lattice_array, tmpR1, m)
                    #We found a bead
                    if new_bead != -1:
                        new_chain = topo_array[TOPO_ARRAY_MAX * new_bead + TOPO_ARRAY_CHAIN]
                        #The bead belongs to a new chain
                        if cur_clust_checklist[new_chain] != 1:
                            cluster_list[clus_size] = new_chain
                            clus_size += 1
                            cur_clust_checklist[new_chain] = 1

            chain_checklist[cur_chain] = 1
        
        if clus_size > cur_largest:
            cur_largest = clus_size
            for j in range(total_num_chains[0]):
                cluster_list_largest[j] = cluster_list[j]
        
        cluster_size_dilute_array[clus_size - 1] += 1
    
    cluster_size_dilute_array[cur_largest - 1] -= 1
    cluster_size_dense_array[cur_largest - 1] += 1

    free(cluster_list)
    free(cur_clust_checklist)
    free(chain_checklist)
    free(cluster_size_list)

    return cur_largest


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cdef Scaling_Analysis(int *frame_array, long double *scaling_array_clus,
                      long double *scaling_array_solo, const int *cluster_list_largest,
                      const int clus_size, const int *chain_id_to_distinct_chain_array,
                      const int *chain_id_to_first_bead_array, const int *chain_length_array,
                      const int *scaling_length_fb_array):
     
    cdef:
        int i, j, k
        int fb, lb
        int counter = 0
        int *clus_checklist
 
    clus_checklist = <int *> calloc(total_num_chains[0], sizeof(int))
     
    for i in range(clus_size):
        clus_checklist[cluster_list_largest[i]] = 1
     
    for i in range(total_num_chains[0]):
        fb = chain_id_to_first_bead_array[i]
        lb = fb + chain_length_array[chain_id_to_distinct_chain_array[i]]
        scaling_array_fb = scaling_length_fb_array[chain_id_to_distinct_chain_array[i]]
        if clus_checklist[i] == 1:
            for j in range(fb, lb-1):
                for k in range(j+1, lb):
                    scaling_array_clus[scaling_array_fb + counter] += Dist_Bead_Bead_Squared_No_PBC(frame_array, j, k)
                    counter += 1
            counter = 0
        else:
            for j in range(fb, lb-1):
                for k in range(j+1, lb):
                    scaling_array_solo[scaling_array_fb + counter] += Dist_Bead_Bead_Squared(frame_array, j, k)
                    counter += 1
            counter = 0
 
    free(clus_checklist)

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cdef Rg_Full_Analysis(long double *rg_array, long double *rg_full_array,
                      long *rg_full_count_array,
                      const int *cluster_list_largest, const int clus_size,
                      const int *chain_id_to_distinct_chain_array):
     
    cdef:
        int i
        int chain_type
        int *clus_checklist
 
    clus_checklist = <int *> calloc(total_num_chains[0], sizeof(int))
     
    for i in range(clus_size):
        clus_checklist[cluster_list_largest[i]] = 1
     
    for i in range(total_num_chains[0]):
        chain_type = chain_id_to_distinct_chain_array[i]
        if clus_checklist[i] == 1:
            rg_full_array[chain_type * RG_FULL_MAX + RG_FULL_DEN] += rg_array[i]
            rg_full_count_array[chain_type * RG_FULL_MAX + RG_FULL_DEN] += 1
        else:
            rg_full_array[chain_type * RG_FULL_MAX + RG_FULL_DIL] += rg_array[i]
            rg_full_count_array[chain_type * RG_FULL_MAX + RG_FULL_DIL] += 1
 
    free(clus_checklist)

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cdef End_To_End_Analysis(int *frame_array, const int *cluster_list_largest,
                         const int clus_size, const int *chain_id_to_distinct_chain_array,
                         const int *chain_id_to_first_bead_array,
                         const int *chain_length_array, char* re_file_clus,
                         char* re_file_solo):
     
    cdef:
        int i
        int fb, lb
        int *clus_checklist
 
    clus_checklist = <int *> calloc(total_num_chains[0], sizeof(int))
     
    for i in range(clus_size):
        clus_checklist[cluster_list_largest[i]] = 1
     
    for i in range(total_num_chains[0]):
        fb = chain_id_to_first_bead_array[i]
        lb = fb + chain_length_array[chain_id_to_distinct_chain_array[i]]
        if clus_checklist[i] == 1:
            re_dist_array_clus[2 * re_count_clus[0]] = chain_id_to_distinct_chain_array[i]
            re_dist_array_clus[2 * re_count_clus[0] + 1] = Dist_Bead_Bead_Squared_No_PBC(frame_array, fb, lb)
            re_count_clus[0] += 1
            if re_count_clus[0] == RE_DIST_ARRAY_SIZE:
                Re_Dist_Writer_Clus(re_file_clus)
        else:
            re_dist_array_solo[2 * re_count_solo[0]] = chain_id_to_distinct_chain_array[i]
            re_dist_array_solo[2 * re_count_solo[0] + 1] = Dist_Bead_Bead_Squared(frame_array, fb, lb)
            re_count_solo[0] += 1
            if re_count_solo[0] == RE_DIST_ARRAY_SIZE:
                Re_Dist_Writer_Solo(re_file_solo)
     
    free(clus_checklist)

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cdef Gyration_Tensor_Cluster_Analysis(int *frame_array, int *cluster_list_largest,
                                      const int clus_size,
                                      const int *chain_id_to_distinct_chain_array,
                                      const int *chain_id_to_first_bead_array,
                                      const int *chain_length_array, char* gtensor_file):
    cdef:
        int i, j, k
        int c1, b1, b2
        int num_clus_beads = 0
        int bead_counter = 0
        int *clus_beads_array
        
         
    for i in range(clus_size):
        num_clus_beads += chain_length_array[chain_id_to_distinct_chain_array[cluster_list_largest[i]]]
    
    clus_beads_array = <int *> malloc(num_clus_beads * sizeof(int))
    for i in range(clus_size):
        c1 = cluster_list_largest[i]
        b1 = chain_id_to_first_bead_array[c1]
        b2 = b1 + chain_length_array[chain_id_to_distinct_chain_array[c1]]
        for j in range(b1, b2):
            clus_beads_array[bead_counter] = j
            bead_counter += 1
    
    for i in range(num_clus_beads - 1):
        b1 = clus_beads_array[i]
        for j in range(i + 1, num_clus_beads):
            b2 = clus_beads_array[j]
            for k in range(3):
                gtensor_dist_array[gtensor_count[0] * 6 + k] += <long double> (Dist_Bead_Bead_1D_No_PBC(frame_array, b1, b2, k) * Dist_Bead_Bead_1D_No_PBC(frame_array, b1, b2, k)) / <long double> (num_clus_beads * num_clus_beads)
                gtensor_dist_array[gtensor_count[0] * 6 + k + 3] += <long double> (Dist_Bead_Bead_1D_No_PBC(frame_array, b1, b2, k) * Dist_Bead_Bead_1D_No_PBC(frame_array, b1, b2, (k + 1) % 3)) / <long double> (num_clus_beads * num_clus_beads)
    gtensor_count[0] += 1
     
    if gtensor_count[0] == GYRATION_TENSOR_DIST_ARRAY_SIZE:
        Gtensor_Dist_Writer(gtensor_file)
    
    free(clus_beads_array)

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cdef Fractal_Analysis(int *frame_array, int *topo_array, int *lattice_array, int *system_COM, int *sticker_array, char* fractal_ana_file):
    cdef:
        int i, j, k, x, y, z
        int temp_B, new_bead
        int fractal_center[3]
        int tmpR1[3]
        long double rand_val
        long double dist_COM_fractal
     
    for i in range(fractal_ana_samples[0]):
        dist_COM_fractal = 0
        for j in range(3):
            rand_val = <long double> (rand() / (RAND_MAX + 1.0))
            fractal_center[j] = <int> (2 * fractal_ana_max_distance[0] * rand_val) + system_COM[j] - fractal_ana_max_distance[0]
            fractal_ana_dist_array[fractal_ana_count[0] * FRACTAL_MAX + j] = fractal_center[j]
            dist_COM_fractal += (fractal_center[j] - system_COM[j]) * (fractal_center[j] - system_COM[j])
        fractal_ana_dist_array[fractal_ana_count[0] * FRACTAL_MAX + FRACTAL_DISTANCE] = sqrt(dist_COM_fractal)
         
        for x in range(fractal_center[0] - fractal_ana_radius[0], fractal_center[0] + fractal_ana_radius[0] + 1):
            if x < 0:
                tmpR1[0] = x + box_size[0]
            elif x >= box_size[0]:
                tmpR1[0] = x - box_size[0]
            else:
                tmpR1[0] = x
            for y in range(fractal_center[1] - fractal_ana_radius[0], fractal_center[1] + fractal_ana_radius[0] + 1):
                if y < 0:
                    tmpR1[1] = y + box_size[0]
                elif y >= box_size[0]:
                    tmpR1[1] = y - box_size[0]
                else:
                    tmpR1[1] = y
                for z in range(fractal_center[2] - fractal_ana_radius[0], fractal_center[2] + fractal_ana_radius[0] + 1):
                    if z < 0:
                        tmpR1[2] = z + box_size[0]
                    elif z >= box_size[0]:
                        tmpR1[2] = z - box_size[0]
                    else:
                        tmpR1[2] = z
                    temp_B = lattice_array[box_size[0] * box_size[0] * tmpR1[0] + box_size[0] * tmpR1[1] + tmpR1[2]]
                    if temp_B == -1:
                        fractal_ana_dist_array[fractal_ana_count[0] * FRACTAL_MAX + FRACTAL_EMPTY] += 1
                    else:
                        for j in range(num_stickers[0]):
                            if topo_array[temp_B * TOPO_ARRAY_MAX + TOPO_ARRAY_BEAD] == sticker_array[j]:
                                fractal_ana_dist_array[fractal_ana_count[0] * FRACTAL_MAX + FRACTAL_STICKER] += 1
                                for k in range(27):
                                    if k == 13:
                                        continue
                                    new_bead = Rotation_Check(lattice_array, tmpR1, k)
                                    #We found a bead
                                    if new_bead == -1:
                                        fractal_ana_dist_array[fractal_ana_count[0] * FRACTAL_MAX + FRACTAL_EMPTY_NEIGHBOR] += 1
                                    else:
                                        for m in range(num_stickers[0]):
                                            if topo_array[TOPO_ARRAY_MAX * new_bead + TOPO_ARRAY_BEAD] == sticker_array[m]:
                                                fractal_ana_dist_array[fractal_ana_count[0] * FRACTAL_MAX + FRACTAL_STICKER_NEIGHBOR] += 1
                                                break
                                        else:
                                            fractal_ana_dist_array[fractal_ana_count[0] * FRACTAL_MAX + FRACTAL_SPACER_NEIGHBOR] += 1
                                continue
 
        fractal_ana_count[0] += 1
        if fractal_ana_count[0] == FRACTAL_ANA_DIST_ARRAY_SIZE:
            Fractal_Ana_Dist_Writer(fractal_ana_file)

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cdef mol_COM_Analysis(int *frame_array, int *topo_array, long double *mol_COM_array,
                      long double *mol_COM_array_old, long double *mol_COM_binned_array,
                      long *mol_COM_binned_count_array, int *system_COM,
                      int *cluster_list_largest, int clus_size,
                      const int *chain_id_to_distinct_chain_array,
                      const int *chain_id_to_first_bead_array,
                      const int *chain_length_array, char *displacement_file):
 
    cdef:
        int i, j, cur_bead, cur_chain_length
        int chain_type
        int boundary_check[3]
        int *clus_checklist
        long double cur_bin
        long double dist_sq
        long double cur_dist
        long double com_dist
        long double mol_COM[3]
 
    clus_checklist = <int *> calloc(total_num_chains[0], sizeof(int))
     
    for i in range(clus_size):
        clus_checklist[cluster_list_largest[i]] = 1
             
    cur_bead = 0
    for i in range(total_num_chains[0]):
        cur_chain_length = 0
        for j in range(3):
            mol_COM[j] = 0
            boundary_check[j] = 0
        while topo_array[cur_bead * TOPO_ARRAY_MAX + TOPO_ARRAY_CHAIN] == i:
            for j in range(3):
                if cur_chain_length > 0:
                    if (frame_array[cur_bead * FRAME_ARRAY_MAX + j] - frame_array[(cur_bead - 1) * FRAME_ARRAY_MAX + j]) > (box_size[0] / 2):
                        boundary_check[j] -= 1
                    elif (frame_array[cur_bead * FRAME_ARRAY_MAX + j] - frame_array[(cur_bead - 1) * FRAME_ARRAY_MAX + j]) < - (box_size[0] / 2):
                        boundary_check[j] += 1
                mol_COM[j] += frame_array[cur_bead * FRAME_ARRAY_MAX + j] + boundary_check[j] * box_size[0]
            cur_chain_length += 1
            cur_bead += 1
            if cur_bead == num_beads[0]:
                break
 
        for j in range(3):
            mol_COM[j] /= cur_chain_length
            if mol_COM[j] < 0:
                mol_COM[j] = mol_COM[j] + box_size[0]
            elif mol_COM[j] >= box_size[0]:
                mol_COM[j] = mol_COM[j] - box_size[0]
            mol_COM_array[i * 4 + j] = mol_COM[j]
        mol_COM_array[i * 4 + 3] = clus_checklist[i]
     
    if mol_COM_array_old[0] + mol_COM_array_old[1] + mol_COM_array_old[2] + mol_COM_array_old[3] > 0:
        for i in range(total_num_chains[0]):
            cur_bin = 0
            dist_sq = 0
            chain_type = chain_id_to_distinct_chain_array[i]
            for j in range(3):
                com_dist = abs(system_COM[j] - mol_COM_array_old[i * 4 + j])
                if com_dist > box_size[0] / 2:
                    com_dist = box_size[0] - com_dist 
                cur_bin += com_dist ** 2               
                cur_dist = abs(mol_COM_array[i * 4 + j] - mol_COM_array_old[i * 4 + j])
                if cur_dist > box_size[0] / 2:
                    cur_dist = box_size[0] - cur_dist
                dist_sq += cur_dist ** 2
            cur_bin = <int> (4 * sqrt(cur_bin))
            mol_COM_binned_array[<int> (cur_bin * num_distinct_chains[0] + chain_type)] += <long double> sqrt(dist_sq)
            mol_COM_binned_count_array[<int> (cur_bin * num_distinct_chains[0] + chain_type)] += 1
            
            displacement_skip_count[0] += 1
            if displacement_skip_count[0] == DISPLACEMENT_SKIP:
                displacement_array[3 * displacement_count[0]] = 4 * sqrt(cur_bin)
                displacement_array[3 * displacement_count[0] + 1] = chain_type
                displacement_array[3 * displacement_count[0] + 2] = dist_sq
                displacement_count[0] += 1
                if displacement_count[0] == DISPLACEMENT_ANA_DIST_ARRAY_SIZE:
                    Displacement_Writer(displacement_file)
                displacement_skip_count[0] = 0
 
    free(clus_checklist)

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cdef Sticker_Cluster_Analysis(int *frame_array, int *topo_array, int *lattice_array,
                              int *sticker_array, const int *cluster_list_largest,
                              const int clus_size_largest, int *sticker_cluster_array,
                              const int *chain_id_to_distinct_chain_array,
                              const int *chain_id_to_first_bead_array,
                              const int *chain_length_array):
 
    cdef:
        int i, j, k, m
        int cur_bead
        int clus_size
        int new_bead
        int tmpR1[3]
        int *cluster_list
        int *cur_clust_checklist
        int *bead_checklist
 
    cluster_list = <int *> malloc(num_beads[0] * sizeof(int))
    cur_clust_checklist = <int *> malloc(num_beads[0] * sizeof(int))
    bead_checklist = <int *> calloc(num_beads[0], sizeof(int))
 
    for i in range(num_beads[0]):
         
        if bead_checklist[i] == 1:
            continue
         
        for j in range(clus_size_largest):
            if topo_array[TOPO_ARRAY_MAX * i + TOPO_ARRAY_CHAIN] == cluster_list_largest[j]:
                break
        else:
            bead_checklist[i] = 1
            continue
         
        for j in range(num_stickers[0]):
            if topo_array[TOPO_ARRAY_MAX * i + TOPO_ARRAY_BEAD] == sticker_array[j]:
                break
        else:
            bead_checklist[i] = 1
            continue
         
        for j in range(num_beads[0]):
            cluster_list[j] = -1
            cur_clust_checklist[j] = 0
 
        cur_clust_checklist[i] = 1
        cluster_list[0] = i
        clus_size = 1
         
        for j in range(num_beads[0]):
            cur_bead = cluster_list[j]
            if cur_bead == -1:
                break
            for m in range(3):
                tmpR1[m] = frame_array[FRAME_ARRAY_MAX * cur_bead + m]
            for m in range(27):
                if m == 13:
                    continue
                new_bead = Rotation_Check(lattice_array, tmpR1, m)
                #We found a bead
                if new_bead != -1:
                     
                    for k in range(clus_size_largest):
                        if topo_array[TOPO_ARRAY_MAX * new_bead + TOPO_ARRAY_CHAIN] == cluster_list_largest[k]:
                            break
                    else:
                        bead_checklist[new_bead] = 1
                        continue
                     
                    for k in range(num_stickers[0]):
                        if topo_array[TOPO_ARRAY_MAX * new_bead + TOPO_ARRAY_BEAD] == sticker_array[k]:
                            break
                    else:
                        bead_checklist[new_bead] = 1
                        continue
                     
                    if cur_clust_checklist[new_bead] != 1:
                        cluster_list[clus_size] = new_bead
                        clus_size += 1
                        cur_clust_checklist[new_bead] = 1
             
            bead_checklist[cur_bead] = 1
         
        sticker_cluster_array[clus_size] += 1
        total_num_cluster_stickers[0] += clus_size
 
    free(cluster_list)
    free(cur_clust_checklist)
    free(bead_checklist)

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cdef Crosslink_Analysis(int *frame_array, int *topo_array, int *lattice_array, int *sticker_array, long double *crosslink_ana_chain_array,
                        long double *crosslink_ana_full_array, long *crosslink_ana_full_count_array, 
                        const int *cluster_list_largest, const int clus_size_largest,
                        int *chain_id_to_distinct_chain_array, int *chain_length_array):

    cdef:
        int i, j, k
        int bead_type_1, bead_type_2
        int chain_1, chain_2
        int chain_type_1, chain_type_2
        int sticker_check_1, sticker_check_2
        int cluster_check_1
        int new_bead
        int tot_clus_beads = 0
        int *tot_type_interactions_array
        int *clus_beads_array
        long double *intermol_cluster_interactions_array
        int tmpR1[3]
    
    tot_type_interactions_array = <int *> calloc(num_distinct_chains[0], sizeof(int))
    clus_beads_array = <int *> calloc(num_distinct_chains[0], sizeof(int))
    intermol_cluster_interactions_array = <long double *> calloc(num_distinct_chains[0] * num_distinct_chains[0], sizeof(long double))

    for i in range(clus_size_largest):
        chain_type_1 = chain_id_to_distinct_chain_array[cluster_list_largest[i]]
        clus_beads_array[chain_type_1] += chain_length_array[chain_type_1]
    
    for i in range(num_distinct_chains[0]):
        tot_clus_beads += clus_beads_array[i]
    
    for i in range(num_beads[0]):
        sticker_check_1 = 0
        cluster_check_1 = 0
        chain_1 = topo_array[TOPO_ARRAY_MAX * i + TOPO_ARRAY_CHAIN]
        chain_type_1 = chain_id_to_distinct_chain_array[chain_1]
        bead_type_1 = topo_array[TOPO_ARRAY_MAX * i + TOPO_ARRAY_BEAD]

        for j in range(num_stickers[0]):
            if bead_type_1 == sticker_array[j]:
                sticker_check_1 = 1
                break
        
        for j in range(clus_size_largest):
            if chain_1 == cluster_list_largest[j]:
                cluster_check_1 = 1
                break

        if cluster_check_1 == 1 and sticker_check_1 == 1:
            crosslink_count_array[CROSSLINK_COUNT_CLUS_ST] += 1
        elif cluster_check_1 == 1:
            crosslink_count_array[CROSSLINK_COUNT_CLUS_SP] += 1
        elif sticker_check_1 == 1:
            crosslink_count_array[CROSSLINK_COUNT_DIL_ST] += 1
        else:
            crosslink_count_array[CROSSLINK_COUNT_DIL_SP] += 1
        
        if cluster_check_1 == 1:
            crosslink_ana_full_count_array[CROSSLINK_FULL_COUNT_CLUS * bead_types[0] + bead_type_1] += 1
        else:
            crosslink_ana_full_count_array[CROSSLINK_FULL_COUNT_DIL * bead_types[0] + bead_type_1] += 1
        
        for j in range(3):
            tmpR1[j] = frame_array[FRAME_ARRAY_MAX * i + j]
                        
        for j in range(27):
            if j == 13:
                continue

            new_bead = Rotation_Check(lattice_array, tmpR1, j)

            if new_bead == -1:
                if cluster_check_1 == 1 and sticker_check_1 == 1:
                    crosslink_ana_array[CROSSLINK_CLUS_ST_SO] += 1
                elif cluster_check_1 == 1:
                    crosslink_ana_array[CROSSLINK_CLUS_SP_SO] += 1
                elif sticker_check_1 == 1:
                    crosslink_ana_array[CROSSLINK_DIL_ST_SO] += 1
                else:
                    crosslink_ana_array[CROSSLINK_DIL_SP_SO] += 1
                if cluster_check_1 == 1:
                    crosslink_ana_full_array[CROSSLINK_FULL_CLUS * bead_types[0] * (bead_types[0] + 1) + bead_type_1 * (bead_types[0] + 1) + bead_types[0]] += 1
                    crosslink_ana_full_array[CROSSLINK_FULL_CLUS_CHAIN * bead_types[0] * (bead_types[0] + 1) + bead_type_1 * (bead_types[0] + 1) + bead_types[0]] += 1
                else:
                    crosslink_ana_full_array[CROSSLINK_FULL_DIL * bead_types[0] * (bead_types[0] + 1) + bead_type_1 * (bead_types[0] + 1) + bead_types[0]] += 1
                    crosslink_ana_full_array[CROSSLINK_FULL_DIL_CHAIN * bead_types[0] * (bead_types[0] + 1) + bead_type_1 * (bead_types[0] + 1) + bead_types[0]] += 1
            else:
                chain_2 = topo_array[TOPO_ARRAY_MAX * new_bead + TOPO_ARRAY_CHAIN]
                chain_type_2 = chain_id_to_distinct_chain_array[chain_2]
                bead_type_2 = topo_array[TOPO_ARRAY_MAX * new_bead + TOPO_ARRAY_BEAD]
                sticker_check_2 = 0
                for k in range(num_stickers[0]):
                    if bead_type_2 == sticker_array[k]:
                        sticker_check_2 = 1
            
                if sticker_check_2 == 1:
                    if cluster_check_1 == 1 and sticker_check_1 == 1:
                        crosslink_ana_array[CROSSLINK_CLUS_ST_ST] += 1
                    elif cluster_check_1 == 1:
                        crosslink_ana_array[CROSSLINK_CLUS_SP_ST] += 1
                    elif sticker_check_1 == 1:
                        crosslink_ana_array[CROSSLINK_DIL_ST_ST] += 1
                    else:
                        crosslink_ana_array[CROSSLINK_DIL_SP_ST] += 1
                    
                    if chain_1 != chain_2:
                        if cluster_check_1 == 1 and sticker_check_1 == 1:
                            crosslink_ana_array[CROSSLINK_CLUS_ST_ST_CHAIN] += 1
                        elif cluster_check_1 == 1:
                            crosslink_ana_array[CROSSLINK_CLUS_SP_ST_CHAIN] += 1

                    if chain_1 != chain_2 or (new_bead != i + 1 and new_bead != i - 1):
                        if cluster_check_1 == 1 and sticker_check_1 == 1:
                            crosslink_ana_array[CROSSLINK_CLUS_ST_ST_CON] += 1
                        elif cluster_check_1 == 1:
                            crosslink_ana_array[CROSSLINK_CLUS_SP_ST_CON] += 1
                
                else:
                    if cluster_check_1 == 1 and sticker_check_1 == 1:
                        crosslink_ana_array[CROSSLINK_CLUS_ST_SP] += 1
                    elif cluster_check_1 == 1:
                        crosslink_ana_array[CROSSLINK_CLUS_SP_SP] += 1
                    elif sticker_check_1 == 1:
                        crosslink_ana_array[CROSSLINK_DIL_ST_SP] += 1
                    else:
                        crosslink_ana_array[CROSSLINK_DIL_SP_SP] += 1                    
                    
                    if chain_1 != chain_2:
                        if cluster_check_1 == 1 and sticker_check_1 == 1:
                            crosslink_ana_array[CROSSLINK_CLUS_ST_SP_CHAIN] += 1
                        elif cluster_check_1 == 1:
                            crosslink_ana_array[CROSSLINK_CLUS_SP_SP_CHAIN] += 1

                    if chain_1 != chain_2 or (new_bead != i + 1 and new_bead != i - 1):
                        if cluster_check_1 == 1 and sticker_check_1 == 1:
                            crosslink_ana_array[CROSSLINK_CLUS_ST_SP_CON] += 1
                        elif cluster_check_1 == 1:
                            crosslink_ana_array[CROSSLINK_CLUS_SP_SP_CON] += 1
                
                if cluster_check_1 == 1:
                    crosslink_ana_full_array[CROSSLINK_FULL_CLUS * bead_types[0] * (bead_types[0] + 1) + bead_type_1 * (bead_types[0] + 1) + bead_type_2] += 1
                    if chain_1 != chain_2:
                        crosslink_ana_full_array[CROSSLINK_FULL_CLUS_CHAIN * bead_types[0] * (bead_types[0] + 1) + bead_type_1 * (bead_types[0] + 1) + bead_type_2] += 1                        
                        if chain_type_1 == chain_type_2:
                            intermol_cluster_interactions_array[chain_type_1 * num_distinct_chains[0] + chain_type_2] += <long double> (tot_clus_beads - chain_length_array[chain_type_1]) / <long double> (clus_beads_array[chain_type_2] - chain_length_array[chain_type_1])
                        else:
                            intermol_cluster_interactions_array[chain_type_1 * num_distinct_chains[0] + chain_type_2] += <long double> (tot_clus_beads - chain_length_array[chain_type_1]) / <long double> (clus_beads_array[chain_type_2])
                        tot_type_interactions_array[chain_type_1] += 1
                        
                else:
                    crosslink_ana_full_array[CROSSLINK_FULL_DIL * bead_types[0] * (bead_types[0] + 1) + bead_type_1 * (bead_types[0] + 1) + bead_type_2] += 1
                    if chain_1 != chain_2:
                        crosslink_ana_full_array[CROSSLINK_FULL_DIL_CHAIN * bead_types[0] * (bead_types[0] + 1) + bead_type_1 * (bead_types[0] + 1) + bead_type_2] += 1
    
    for i in range(num_distinct_chains[0]):
        for j in range(num_distinct_chains[0]):
            if tot_type_interactions_array[i] != 0:
                intermol_cluster_interactions_array[i * num_distinct_chains[0] + j] /= <long double> tot_type_interactions_array[i]
            crosslink_ana_chain_array[i * num_distinct_chains[0] + j] += intermol_cluster_interactions_array[i * num_distinct_chains[0] + j]

    free(tot_type_interactions_array)
    free(clus_beads_array)
    free(intermol_cluster_interactions_array)

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cdef Connectedness_Analysis(int *frame_array, int *topo_array, int *lattice_array, int *sticker_array,
                            long double *connectedness_array,
                            long double *connectedness_binned_array, long *connectedness_binned_count_array,
                            long double *connectedness_full_array,
                            long double *connectedness_betweenness_array, long double *connectedness_betweenness_array_old,
                            long double *connectedness_mol_COM_array,
                            long double *connectedness_mol_COM_array_old, int *connectedness_sticker_array,
                            int *connectedness_sticker_array_old, long double *cluster_COM,
                            int *chain_id_to_distinct_chain_array, int *chain_id_to_first_bead_array,
                            int *chain_length_array, char *connectedness_file, char *connectedness_full_file):

    cdef:
        int i, j, k, m, n
        int connect_index
        int fb, lb
        int chain_type
        int chain_length
        int cur_chain
        int new_chain
        int new_bead
        int bead_type_1, bead_type_2
        int cur_bin
        int sticker_count
        int old_check = 0
        int boundary_check[3]
        int tmpR1[3]
        long double betweenness
        long double dist_sq
        long double cur_dist
        long double mol_COM[3]
        int *degree_checklist
        int *top_betweenness_chains
        long double *top_betweenness

    degree_checklist = <int *> calloc(total_num_chains[0] * total_num_chains[0], sizeof(int))
    top_betweenness_chains = <int *> malloc(connectedness_chains[0] * num_distinct_chains[0] * sizeof(int))
    top_betweenness = <long double *> malloc(connectedness_chains[0] * num_distinct_chains[0] * sizeof(long double))
    
    for i in range(total_num_chains[0] * 3):
        connectedness_mol_COM_array[i] = 0
        
    for i in range(total_num_chains[0]):
        connectedness_betweenness_array[i] = 0
        connectedness_sticker_array[i] = 0
    
    for i in range(connectedness_chains[0] * num_distinct_chains[0]):
        top_betweenness_chains[i] = -1
        top_betweenness[i] = 0
        
    for i in range(total_num_chains[0]):
        chain_type = chain_id_to_distinct_chain_array[i]
        chain_length = chain_length_array[chain_type]
        fb = chain_id_to_first_bead_array[i]
        lb = fb + chain_length
        for j in range(3):
            mol_COM[j] = 0
            boundary_check[j] = 0
        for j in range(fb, lb):
            for k in range(3):
                if j > fb:
                    if (frame_array[j * FRAME_ARRAY_MAX + k] - frame_array[(j - 1) * FRAME_ARRAY_MAX + k]) > (box_size[0] / 2):
                        boundary_check[k] -= 1
                    elif (frame_array[j * FRAME_ARRAY_MAX + k] - frame_array[(j - 1) * FRAME_ARRAY_MAX + k]) < - (box_size[0] / 2):
                        boundary_check[k] += 1
                mol_COM[k] += frame_array[j * FRAME_ARRAY_MAX + k] + boundary_check[k] * box_size[0]
                tmpR1[k] = frame_array[j * FRAME_ARRAY_MAX + k]
            
            bead_type_1 = topo_array[j * TOPO_ARRAY_MAX + TOPO_ARRAY_BEAD]
            for k in range(num_stickers[0]):
                if bead_type_1 == sticker_array[k]:
                    for m in range(27):
                        if m == 13:
                            continue
            
                        new_bead = Rotation_Check(lattice_array, tmpR1, m)
                        bead_type_2 = topo_array[new_bead * TOPO_ARRAY_MAX + TOPO_ARRAY_BEAD]
                        
                        if new_bead != -1:
                            for n in range(num_stickers[0]):
                                if bead_type_2 == sticker_array[n]:
                                    new_chain = topo_array[TOPO_ARRAY_MAX * new_bead + TOPO_ARRAY_CHAIN]
                                    if new_chain != i:
                                        connectedness_sticker_array[i] += 1
                                        if degree_checklist[i * total_num_chains[0] + new_chain] == 0:
                                            degree_checklist[i * total_num_chains[0] + new_chain] = 1
                                    break
                    break

        for j in range(3):
            mol_COM[j] /= chain_length
            if mol_COM[j] < 0:
                mol_COM[j] = mol_COM[j] + box_size[0]
            elif mol_COM[j] >= box_size[0]:
                mol_COM[j] = mol_COM[j] - box_size[0]
            connectedness_mol_COM_array[i * 3 + j] = mol_COM[j]
        
    Betweenness_Calculator(degree_checklist, connectedness_betweenness_array)
        
    if connectedness_skip_count[0] == -1:
        for i in range(3 * total_num_chains[0]):
            connectedness_mol_COM_array_old[i] = connectedness_mol_COM_array[i]
        
        for i in range(total_num_chains[0]):
            connectedness_betweenness_array_old[i] = connectedness_betweenness_array[i]
            connectedness_sticker_array_old[i] = connectedness_sticker_array[i]
        
        connectedness_skip_count[0] = 0
    
    elif connectedness_skip_count[0] == CONNECTEDNESS_SKIP:
        for i in range(total_num_chains[0]):
            chain_type = chain_id_to_distinct_chain_array[i]
            betweenness = connectedness_betweenness_array_old[i]
            sticker_count = connectedness_sticker_array_old[i]
            dist_sq = 0
            for j in range(3):
                cur_dist = abs(connectedness_mol_COM_array[i * 3 + j] - connectedness_mol_COM_array_old[i * 3 + j])
                if cur_dist > box_size[0] / 2:
                    cur_dist = box_size[0] - cur_dist
                dist_sq += cur_dist ** 2
            if betweenness == 0:
                cur_bin = 0
            elif betweenness <= 1:
                cur_bin = 1
            elif betweenness <= 5:
                cur_bin = 2
            elif betweenness <= 10:
                cur_bin = 3
            elif betweenness <= 50:
                cur_bin = 4
            elif betweenness <= 100:
                cur_bin = 5
            elif betweenness <= 500:
                cur_bin = 6
            elif betweenness <= 1000:
                cur_bin = 7
            elif betweenness <= 5000:
                cur_bin = 8
            else:
                cur_bin = BETWEENNESS_BINS - 1
            connectedness_binned_array[cur_bin * num_distinct_chains[0] + chain_type] += dist_sq
            connectedness_binned_count_array[cur_bin * num_distinct_chains[0] + chain_type] += 1
            connectedness_full_array[connectedness_full_count[0] * total_num_chains[0] * 3 + i * 3] = betweenness
            connectedness_full_array[connectedness_full_count[0] * total_num_chains[0] * 3 + i * 3 + 1] = sticker_count
            connectedness_full_array[connectedness_full_count[0] * total_num_chains[0] * 3 + i * 3 + 2] = dist_sq
        
        connectedness_full_count[0] += 1
        if connectedness_full_count[0] == CONNECTEDNESS_FULL_ARRAY_SIZE:
            Connectedness_Full_Writer(connectedness_full_file, connectedness_full_array)
    
        for i in range(3 * total_num_chains[0]):
            connectedness_mol_COM_array_old[i] = connectedness_mol_COM_array[i]
        
        for i in range(total_num_chains[0]):
            connectedness_betweenness_array_old[i] = connectedness_betweenness_array[i]
            connectedness_sticker_array_old[i] = connectedness_sticker_array[i]
        
        connectedness_skip_count[0] = 0
    
    else:
        connectedness_skip_count[0] += 1
    
    for i in range(total_num_chains[0]):
        chain_type = chain_id_to_distinct_chain_array[i]
        betweenness = connectedness_betweenness_array[i]
        for j in range(connectedness_chains[0]):
            if betweenness > top_betweenness[chain_type * connectedness_chains[0] + j]:
                for k in range(connectedness_chains[0] - j - 1):
                    connect_index = chain_type * connectedness_chains[0] + connectedness_chains[0] - k - 1
                    top_betweenness[connect_index] = top_betweenness[connect_index - 1]
                    top_betweenness_chains[connect_index] = top_betweenness_chains[connect_index - 1]
                top_betweenness[chain_type * connectedness_chains[0] + j] = betweenness
                top_betweenness_chains[chain_type * connectedness_chains[0] + j] = i
                break
    
    for i in range(num_distinct_chains[0]):
        for j in range(connectedness_chains[0]):
            cur_chain = top_betweenness_chains[i * connectedness_chains[0] + j]
            connectedness_array[connectedness_count[0] * num_distinct_chains[0] * connectedness_chains[0] * CONNECTEDNESS_MAX + i * connectedness_chains[0] * CONNECTEDNESS_MAX + j * CONNECTEDNESS_MAX + CONNECTEDNESS_CHAIN] = cur_chain
            connectedness_array[connectedness_count[0] * num_distinct_chains[0] * connectedness_chains[0] * CONNECTEDNESS_MAX + i * connectedness_chains[0] * CONNECTEDNESS_MAX + j * CONNECTEDNESS_MAX + CONNECTEDNESS_DEGREES] = top_betweenness[i * connectedness_chains[0] + j]
            connectedness_array[connectedness_count[0] * num_distinct_chains[0] * connectedness_chains[0] * CONNECTEDNESS_MAX + i * connectedness_chains[0] * CONNECTEDNESS_MAX + j * CONNECTEDNESS_MAX + CONNECTEDNESS_X] = connectedness_mol_COM_array[cur_chain * 3] - cluster_COM[0]
            connectedness_array[connectedness_count[0] * num_distinct_chains[0] * connectedness_chains[0] * CONNECTEDNESS_MAX + i * connectedness_chains[0] * CONNECTEDNESS_MAX + j * CONNECTEDNESS_MAX + CONNECTEDNESS_Y] = connectedness_mol_COM_array[cur_chain * 3 + 1] - cluster_COM[1]
            connectedness_array[connectedness_count[0] * num_distinct_chains[0] * connectedness_chains[0] * CONNECTEDNESS_MAX + i * connectedness_chains[0] * CONNECTEDNESS_MAX + j * CONNECTEDNESS_MAX + CONNECTEDNESS_Z] = connectedness_mol_COM_array[cur_chain * 3 + 2] - cluster_COM[2]
    
    connectedness_count[0] += 1
    if connectedness_count[0] == CONNECTEDNESS_ANA_DIST_ARRAY_SIZE:
        Connectedness_Writer(connectedness_file, connectedness_array)
    
    free(degree_checklist)
    free(top_betweenness_chains)
    free(top_betweenness)

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cdef Sticker_Lifetime_Analysis(int *frame_array, int *topo_array, int *lattice_array,
                               int *sticker_array, int *bead_sticker_index_array,
                               int *sticker_lifetime_array, char *sticker_lifetime_file):
    
    cdef:
        int i, j
        int sticker_index_1, sticker_index_2
        int new_bead
        int array_index
        int tmpR1[3]
        int *sticker_sticker_checklist_array
    
    sticker_sticker_checklist_array = <int *> calloc(<int> ((total_num_stickers[0] * total_num_stickers[0] - total_num_stickers[0]) / 2), sizeof(int))
    
    for i in range(num_beads[0]):
        sticker_index_1 = bead_sticker_index_array[i]
        if sticker_index_1 != -1:
            for j in range(3):
                tmpR1[j] = frame_array[FRAME_ARRAY_MAX * i + j]
            for j in range(27):
                if j == 13:
                    continue
                new_bead = Rotation_Check(lattice_array, tmpR1, j)
                if new_bead != -1 and new_bead > i:
                    if not (new_bead == i + 1 and topo_array[TOPO_ARRAY_MAX * i + TOPO_ARRAY_CHAIN] == topo_array[TOPO_ARRAY_MAX * new_bead + TOPO_ARRAY_CHAIN]):
                        sticker_index_2 = bead_sticker_index_array[new_bead]
                        if sticker_index_2 != -1:
                            #Actually psychotic indexing but here we go
                            array_index = <int> (sticker_index_1 * (2 * total_num_stickers[0] - sticker_index_1 - 3) / 2 + sticker_index_2 - 1)
                            sticker_sticker_checklist_array[array_index] = 1
                            sticker_lifetime_array[array_index] += 1
    
    for i in range(<int> ((total_num_stickers[0] * total_num_stickers[0] - total_num_stickers[0]) / 2)):
        if sticker_sticker_checklist_array[i] == 0:
            if sticker_lifetime_array[i] > 0:
                sticker_lifetime_dist_array[sticker_lifetime_count[0]] = sticker_lifetime_array[i]
                sticker_lifetime_array[i] = 0
                sticker_lifetime_count[0] += 1
                if sticker_lifetime_count[0] == STICKER_LIFETIME_ARRAY_SIZE:
                    Sticker_Lifetime_Writer(sticker_lifetime_file,
                                            sticker_lifetime_array, 0)
    
    free(sticker_sticker_checklist_array)
        
@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cdef Concave_Hull_Analysis(const int *frame_array, const long double *cluster_COM,
                           const int *cluster_list_largest, const int clus_size_largest,
                           int *concave_hull_array, long double *concave_hull_com_array,
                           int *chain_id_to_distinct_chain_array,
                           int *chain_id_to_first_bead_array, int *chain_length_array):

    cdef:
        int i, j
        int min_z, max_z
        int cur_chain, chain_type, chain_length
        int b_initial, b_final
        int *lattice_2d_array

    lattice_2d_array = <int *> calloc(box_size[0] * box_size[0], sizeof(int))

    min_z = <int> (cluster_COM[2] - concave_hull_z_thickness[0])
    max_z = <int> (cluster_COM[2] + concave_hull_z_thickness[0])
    for i in range(clus_size_largest):
        cur_chain = cluster_list_largest[i]
        chain_type = chain_id_to_distinct_chain_array[cur_chain]
        chain_length = chain_length_array[chain_type]
        b_initial = chain_id_to_first_bead_array[cur_chain]
        b_final = b_initial + chain_length
        for j in range(b_initial, b_final):
            if min_z <= frame_array[FRAME_ARRAY_MAX * j + FRAME_ARRAY_Z] <= max_z:
                cur_x = frame_array[FRAME_ARRAY_MAX * j + FRAME_ARRAY_X]
                cur_y = frame_array[FRAME_ARRAY_MAX * j + FRAME_ARRAY_Y]
                lattice_2d_array[cur_x * box_size[0] + cur_y] += 1
    
    concave_hull_com_array[count_analysis[0] * 2] = cluster_COM[0]
    concave_hull_com_array[count_analysis[0] * 2 + 1] = cluster_COM[1]

    for i in range(box_size[0]):
        for j in range(box_size[0]):
            if lattice_2d_array[i * box_size[0] + j] >= concave_hull_z_requirement[0]:
                concave_hull_array[count_analysis[0] * box_size[0] * box_size[0] + i * box_size[0] + j] = 1

    free(lattice_2d_array)

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cdef Perform_Analysis(int *frame_array, int *topo_array, int *lattice_array,
                      long double *scaling_array_clus, long double *scaling_array_solo,
                      int *scaling_length_array, int *scaling_length_fb_array,
                      int *chain_length_fb_array, long double *rg_full_array,
                      long *rg_full_count_array,
                      long double *COM_Rg_array, long double *COM_bin_count_array,
                      long *COM_bead_array, long *COM_chain_array, long *COM_count_array,
                      long *COM_sticker_array, long *COM_sticker_count_array,
                      long double *COM_Re_array, long *COM_Re_count_array,
                      long double *COM_angle_array, long *COM_angle_count_array,
                      long double *COM_dot_prod_array, long *COM_dot_prod_count_array,
                      long double *COM_solvation_array, long *COM_solvation_count_array,
                      long double *COM_gtensor_array, long *COM_gtensor_count_array,
                      long long *RDF_array, long double *mol_COM_array,
                      long double *mol_COM_array_old, long double *mol_COM_binned_array,
                      long *mol_COM_binned_count_array, int *sticker_cluster_array,
                      long double *connectedness_array, long double *connectedness_binned_array,
                      long *connectedness_binned_count_array, long double *connectedness_full_array,
                      long double *connectedness_betweenness_array,
                      long double *connectedness_betweenness_array_old,
                      long double *connectedness_mol_COM_array, long double *connectedness_mol_COM_array_old,
                      int *connectedness_sticker_array, int *connectedness_sticker_array_old,
                      int *sticker_lifetime_array,
                      int *cluster_size_dilute_array, int *cluster_size_dense_array,
                      int *sticker_array, int *bead_sticker_index_array,
                      long double *crosslink_ana_chain_array, long double *crosslink_ana_full_array,
                      long *crosslink_ana_full_count_array, int *concave_hull_array,
                      long double *concave_hull_com_array, int *num_chains_array,
                      int *chain_id_to_distinct_chain_array, int *chain_id_to_first_bead_array,
                      int *chain_length_array, int *count_clus_array, int *count_solo_array,
                      char *re_file_clus, char *re_file_solo,
                      char *gtensor_file, char *fractal_ana_file, char *mol_COM_file,
                      char *mol_COM_file_2, char *displacement_file,
                      char *solvation_file, char *connectedness_file, char *connectedness_full_file,
                      char *sticker_lifetime_file):

    cdef:
        int i, j, k
        int clus_size_largest
        int *cluster_list_largest
        int *system_COM
        int *bin_count_array
        int *clus_checklist
        long double *cluster_COM
        long double *rg_array
        long double *asph_acyl_array
        long double *partial_rg_array
        long double *cosine_array

    for i in range(box_size[0]):
        for j in range(box_size[0]):
            for k in range(box_size[0]):
                lattice_array[box_size[0] * box_size[0] * i + box_size[0] * j + k] = -1

    for i in range(num_beads[0]):
        lattice_array[box_size[0] * box_size[0] * frame_array[FRAME_ARRAY_MAX * i + FRAME_ARRAY_X] + box_size[0] * frame_array[FRAME_ARRAY_MAX * i + FRAME_ARRAY_Y] + frame_array[FRAME_ARRAY_MAX * i + FRAME_ARRAY_Z]] = i
         
    cluster_list_largest = <int *> malloc(total_num_chains[0] * sizeof(int))
    clus_size_largest = Cluster_Analysis(frame_array, topo_array, lattice_array, cluster_list_largest,
                                         cluster_size_dilute_array, cluster_size_dense_array,
                                         chain_id_to_distinct_chain_array,
                                         chain_id_to_first_bead_array, chain_length_array)
    clus_checklist = <int *> calloc(total_num_chains[0], sizeof(int))
    
    for i in range(clus_size_largest):
        clus_checklist[cluster_list_largest[i]] = 1
    for i in range(total_num_chains[0]):
        if clus_checklist[i] == 1:
            count_clus_array[chain_id_to_distinct_chain_array[i]] += 1
        else:
            count_solo_array[chain_id_to_distinct_chain_array[i]] += 1
    
    if fractal_ana_dist_check[0] == 1 or com_dist_check[0] == 1 or mol_COM_dist_check[0] == 1 or com_gtensor_dist_check[0] == 1 or connectedness_dist_check[0] == 1 or concave_hull_dist_check[0] == 1:
        system_COM = <int *> malloc(3 * sizeof(int))
        cluster_COM = <long double *> malloc(3 * sizeof(long double))
        COM_Analysis(frame_array, system_COM, cluster_COM, cluster_list_largest,
                     clus_size_largest, chain_id_to_distinct_chain_array,
                     chain_id_to_first_bead_array, chain_length_array)
 
    if com_dist_check[0] == 1 or solvation_dist_check[0] == 1 or nu_check[0] == 1:
        rg_array = <long double *> malloc(total_num_chains[0] * sizeof(long double))
        Rg_Analysis(frame_array, rg_array, cluster_list_largest, clus_size_largest,
                    chain_id_to_distinct_chain_array, chain_id_to_first_bead_array,
                    chain_length_array)

    if nu_check[0] == 1:
        Scaling_Analysis(frame_array, scaling_array_clus, scaling_array_solo,
                         cluster_list_largest, clus_size_largest,
                         chain_id_to_distinct_chain_array,
                         chain_id_to_first_bead_array, chain_length_array,
                         scaling_length_fb_array)
        Rg_Full_Analysis(rg_array, rg_full_array, rg_full_count_array, cluster_list_largest,
                         clus_size_largest, chain_id_to_distinct_chain_array)

    if re_dist_check[0] == 1:
        End_To_End_Analysis(frame_array, cluster_list_largest, clus_size_largest,
                            chain_id_to_distinct_chain_array, chain_id_to_first_bead_array,
                            chain_length_array, re_file_clus, re_file_solo)

    if gtensor_dist_check[0] == 1:
        Gyration_Tensor_Cluster_Analysis(frame_array, cluster_list_largest, clus_size_largest,
                                         chain_id_to_distinct_chain_array,
                                         chain_id_to_first_bead_array, chain_length_array,
                                         gtensor_file)

    if fractal_ana_dist_check[0] == 1:
        Fractal_Analysis(frame_array, topo_array, lattice_array, system_COM, sticker_array, fractal_ana_file)

    if com_dist_check[0] == 1:
        bin_count_array = <int *> malloc(total_num_chains[0] * sizeof(int))
        Bin_Count_Analysis(frame_array, bin_count_array, system_COM,
                           chain_id_to_distinct_chain_array, chain_id_to_first_bead_array,
                           chain_length_array)
        cosine_array = <long double *> malloc(2 * num_beads[0] * sizeof(long double))
        Cosine_Analysis(frame_array, topo_array, cosine_array, system_COM,
                        chain_id_to_distinct_chain_array, chain_length_array)
        COM_Distance_Analysis(frame_array, topo_array, lattice_array, system_COM, COM_Rg_array,
                              COM_bin_count_array, COM_bead_array, COM_chain_array,
                              COM_count_array, COM_sticker_array, COM_sticker_count_array,
                              COM_Re_array, COM_Re_count_array, COM_angle_array, COM_angle_count_array,
                              COM_dot_prod_array, COM_dot_prod_count_array, COM_solvation_array,
                              COM_solvation_count_array, rg_array, bin_count_array, cosine_array, sticker_array,
                              chain_id_to_distinct_chain_array, chain_id_to_first_bead_array,
                              chain_length_array, chain_length_fb_array)

    if RDF_dist_check[0] == 1:
        RDF_Analysis(frame_array, topo_array, RDF_array, sticker_array, cluster_list_largest,
                     clus_size_largest, chain_id_to_distinct_chain_array,
                     chain_id_to_first_bead_array, chain_length_array)

    if mol_COM_dist_check[0] == 1:
        mol_COM_Analysis(frame_array, topo_array, mol_COM_array, mol_COM_array_old,
                         mol_COM_binned_array, mol_COM_binned_count_array,
                         system_COM, cluster_list_largest, clus_size_largest,
                         chain_id_to_distinct_chain_array,
                         chain_id_to_first_bead_array, chain_length_array,
                         displacement_file)
        
        mol_COM_Ana_Dist_Writer(mol_COM_array, mol_COM_array_old, mol_COM_file, mol_COM_file_2)

    if sticker_cluster_dist_check[0] == 1:
        Sticker_Cluster_Analysis(frame_array, topo_array, lattice_array, sticker_array,
                                 cluster_list_largest, clus_size_largest,
                                 sticker_cluster_array, chain_id_to_distinct_chain_array,
                                 chain_id_to_first_bead_array, chain_length_array)
    
    if crosslink_dist_check[0] == 1:
        Crosslink_Analysis(frame_array, topo_array, lattice_array, sticker_array, crosslink_ana_chain_array,
                           crosslink_ana_full_array, crosslink_ana_full_count_array,
                           cluster_list_largest, clus_size_largest, chain_id_to_distinct_chain_array,
                           chain_length_array)
    
    if solvation_dist_check[0] == 1:
        partial_rg_array = <long double *> malloc(PARTIAL_RG_NUM * total_num_chains[0] * sizeof(long double))
        Partial_Rg_Analysis(frame_array, partial_rg_array, cluster_list_largest,
                            clus_size_largest, chain_id_to_distinct_chain_array,
                            chain_id_to_first_bead_array, chain_length_array)
        Solvation_Analysis(frame_array, topo_array, lattice_array, cluster_list_largest,
                           clus_size_largest, rg_array, partial_rg_array, 
                           chain_id_to_distinct_chain_array,
                           chain_id_to_first_bead_array, chain_length_array, solvation_file)

    if com_gtensor_dist_check[0] == 1:
        asph_acyl_array = <long double *> malloc(2 * total_num_chains[0] * sizeof(long double))
        Chain_Asph_Acyl_Calculator(frame_array, asph_acyl_array,
                                   chain_id_to_distinct_chain_array,
                                   chain_id_to_first_bead_array, chain_length_array)
        COM_GTensor_Analysis(frame_array, topo_array, COM_gtensor_array, COM_gtensor_count_array,
                             asph_acyl_array, system_COM, chain_id_to_distinct_chain_array)
        
    if connectedness_dist_check[0] == 1:
        Connectedness_Analysis(frame_array, topo_array, lattice_array, sticker_array,
                               connectedness_array, connectedness_binned_array,
                               connectedness_binned_count_array, connectedness_full_array,
                               connectedness_betweenness_array,
                               connectedness_betweenness_array_old,
                               connectedness_mol_COM_array, connectedness_mol_COM_array_old,
                               connectedness_sticker_array, connectedness_sticker_array_old,
                               cluster_COM, chain_id_to_distinct_chain_array,
                               chain_id_to_first_bead_array, chain_length_array,
                               connectedness_file, connectedness_full_file)
    
    if sticker_lifetime_dist_check[0] == 1:
        Sticker_Lifetime_Analysis(frame_array, topo_array, lattice_array, sticker_array,
                                  bead_sticker_index_array, sticker_lifetime_array,
                                  sticker_lifetime_file)
    
    if concave_hull_dist_check[0] == 1:
        Concave_Hull_Analysis(frame_array, cluster_COM, cluster_list_largest, clus_size_largest,
                              concave_hull_array, concave_hull_com_array,
                              chain_id_to_distinct_chain_array,
                              chain_id_to_first_bead_array, chain_length_array)
        
    count_analysis[0] += 1

    free(cluster_list_largest)
    free(clus_checklist)
    if fractal_ana_dist_check[0] == 1 or com_dist_check[0] == 1 or mol_COM_dist_check[0] == 1 or com_gtensor_dist_check[0] == 1 or connectedness_dist_check[0] == 1 or concave_hull_dist_check[0] == 1:
        free(system_COM)
        free(cluster_COM)
    if com_dist_check[0] == 1 or solvation_dist_check[0] == 1 or nu_check[0] == 1:
        free(rg_array)
    if com_dist_check[0] == 1:
        free(bin_count_array)
        free(cosine_array)
    if solvation_dist_check[0] == 1:
        free(partial_rg_array)
    if com_gtensor_dist_check[0] == 1:
        free(asph_acyl_array)

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cpdef Frame_Loader(const int num_files, const int binary_check, char[:, ::1] trj_file_list,
                   const int trj_file_size, char[:, ::1] topo_file_list,
                   const int topo_file_size, char[:, ::1] nu_file_list,
                   const int nu_size, char[:, ::1] rg_full_file_list,
                   const int rg_full_size, char[:, ::1] re_dist_file_list,
                   const int re_file_size, char[:, ::1] gtensor_dist_file_list,
                   const int gtensor_file_size, char[:, ::1] fractal_ana_dist_file_list,
                   const int fractal_ana_file_size, char[:, ::1] rg_COM_file_list,
                   const int rg_COM_file_size, char[:, ::1] COM_bin_count_file_list,
                   const int COM_bin_count_file_size, char[:, ::1] COM_bead_file_list,
                   const int COM_bead_file_size, char[:, ::1] COM_chain_file_list,
                   const int COM_chain_file_size, char[:, ::1] COM_sticker_file_list,
                   const int COM_sticker_file_size, char[:, ::1] COM_Re_file_list,
                   const int COM_Re_file_size, char[:, ::1] COM_angle_file_list,
                   const int COM_angle_file_size, char[:, ::1] COM_dot_prod_file_list,
                   const int COM_dot_prod_file_size, char[:, ::1] COM_solvation_file_list,
                   const int COM_solvation_file_size, char[:, ::1] RDF_file_list,
                   const int RDF_file_size, char[:, ::1] mol_COM_file_list,
                   const int mol_COM_file_size, char[:, ::1] mol_COM_file_list_2,
                   const int mol_COM_file_size_2, char[:, ::1] mol_COM_file_list_3,
                   const int mol_COM_file_size_3, char[:, ::1] displacement_file_list,
                   const int displacement_file_size, char[:, ::1] sticker_cluster_file_list,
                   const int sticker_cluster_file_size, char[:, ::1] crosslink_file_list,
                   const int crosslink_file_size, char[:, ::1] solvation_file_list,
                   const int solvation_file_size, char[:, ::1] COM_gtensor_file_list,
                   const int COM_gtensor_file_size, char[:, ::1] connectedness_file_list,
                   const int connectedness_file_size, char[:, ::1] connectedness_binned_file_list,
                   const int connectedness_binned_file_size, char[:, ::1] connectedness_full_file_list,
                   const int connectedness_full_file_size, char[:, ::1] sticker_lifetime_file_list,
                   const int sticker_lifetime_file_size, char[:, ::1] concave_hull_file_list,
                   const int concave_hull_file_size, char[:, ::1] cluster_file_list,
                   const int cluster_file_size, const int time_start, const int time_inc,
                   const long time_end, const long time_cutoff, const int nu_check_entry,
                   const int re_dist_check_entry, const int gtensor_dist_check_entry,
                   const int fractal_ana_dist_check_entry, const int com_dist_check_entry,
                   const int RDF_dist_check_entry, const int mol_COM_dist_check_entry,
                   const int sticker_cluster_dist_check_entry, const int crosslink_dist_check_entry,
                   const int solvation_dist_check_entry, const int com_gtensor_dist_check_entry,
                   const int connectedness_dist_check_entry,
                   const int sticker_lifetime_dist_check_entry,
                   const int concave_hull_dist_check_entry, long[:] sticker_array_entry,
                   const int sticker_array_entry_length, const int fractal_ana_samples_entry,
                   const int fractal_ana_radius_entry, const int fractal_ana_max_distance_entry,
                   const int concave_hull_z_thickness_entry, const int concave_hull_z_requirement_entry,
                   const int num_distinct_chains_entry, long[:] num_chains_array_entry):

    cdef:
        int i, j, k
        FILE *fp
        char *cur_file
        char *nu_file
        char *rg_full_file
        char *re_file_clus
        char *re_file_solo
        char *gtensor_file
        char *fractal_ana_file
        char *rg_COM_file
        char *COM_bin_count_file
        char *COM_bead_file
        char *COM_chain_file
        char *COM_sticker_file
        char *COM_Re_file
        char *COM_angle_file
        char *COM_dot_prod_file
        char *COM_solvation_file
        char *RDF_file
        char *mol_COM_file
        char *mol_COM_file_2
        char *mol_COM_file_3
        char *displacement_file
        char *sticker_cluster_file
        char *crosslink_file
        char *cluster_file
        char *solvation_file
        char *COM_gtensor_file
        char *connectedness_file
        char *connectedness_binned_file
        char *connectedness_full_file
        char *sticker_lifetime_file
        char *concave_hull_file
        int *frame_array
        int *topo_array
        int *lattice_array
        int *num_chains_array
        int *chain_id_to_distinct_chain_array
        int *chain_id_to_first_bead_array
        int *chain_length_array
        int *sticker_array
        int *bead_sticker_index_array
        int *scaling_length_array
        int *scaling_length_fb_array
        int *chain_length_fb_array
        int *sticker_cluster_array
        int *connectedness_sticker_array
        int *connectedness_sticker_array_old
        int *sticker_lifetime_array
        int *cluster_size_dilute_array
        int *cluster_size_dense_array
        int *count_clus_array
        int *count_solo_array
        int *concave_hull_array
        long *rg_full_count_array
        long *COM_bead_array
        long *COM_chain_array
        long *COM_count_array
        long *COM_sticker_array
        long *COM_sticker_count_array
        long *COM_Re_count_array
        long *COM_angle_count_array
        long *COM_dot_prod_count_array
        long *COM_solvation_count_array
        long *COM_gtensor_count_array
        long *mol_COM_binned_count_array
        long *connectedness_binned_count_array
        long *crosslink_ana_full_count_array 
        long long *RDF_array
        long double *scaling_array_clus
        long double *scaling_array_solo
        long double *nu_clus_array
        long double *nu_clus_err_array
        long double *nu_solo_array
        long double *nu_solo_err_array
        long double *rg_full_array
        long double *COM_Rg_array
        long double *COM_bin_count_array
        long double *COM_Re_array
        long double *COM_angle_array
        long double *COM_dot_prod_array
        long double *COM_solvation_array
        long double *COM_gtensor_array
        long double *mol_COM_array
        long double *mol_COM_array_old
        long double *mol_COM_binned_array
        long double *connectedness_array
        long double *connectedness_binned_array
        long double *connectedness_full_array
        long double *connectedness_betweenness_array
        long double *connectedness_betweenness_array_old
        long double *connectedness_mol_COM_array
        long double *connectedness_mol_COM_array_old
        long double *crosslink_ana_chain_array
        long double *crosslink_ana_full_array
        long double *concave_hull_com_array
        int skip_frames = 0
        int cur_frame = 0
        int tot_frames = 0
        int topo_check = 0
        int sim_check = 0
        int bead_count = 0
        int bead_type = 0
        int chain_id = 0
        int last_chain_id = -1
        int cur_chain_id = -1
        int chain_type = 0
        int x_pos = 0
        int y_pos = 0
        int z_pos = 0
        int bP = 0
        long line_index
        char strLine[200]
    
    num_beads[0] = 0
    box_size[0] = 0
    total_num_chains[0] = 0
    bead_types[0] = 0
    chain_lengths_sum[0] = 0
    total_scaling_length[0] = 0
    count_analysis[0] = 0
    ana_frames[0] = 0
    re_count_clus[0] = 0
    re_count_solo[0] = 0
    gtensor_count[0] = 0
    solvation_count[0] = 0
    displacement_count[0] = 0
    displacement_skip_count[0] = 0
    fractal_ana_count[0] = 0
    COM_chain_count[0] = 0
    connectedness_count[0] = 0
    connectedness_full_count[0] = 0
    sticker_lifetime_count[0] = 0
    connectedness_skip_count[0] = -1
    total_num_cluster_stickers[0] = 0
    total_num_stickers[0] = 0
    nu_check[0] = nu_check_entry
    re_dist_check[0] = re_dist_check_entry
    gtensor_dist_check[0] = gtensor_dist_check_entry
    fractal_ana_dist_check[0] = fractal_ana_dist_check_entry
    com_dist_check[0] = com_dist_check_entry
    RDF_dist_check[0] = RDF_dist_check_entry
    mol_COM_dist_check[0] = mol_COM_dist_check_entry
    sticker_cluster_dist_check[0] = sticker_cluster_dist_check_entry
    crosslink_dist_check[0] = crosslink_dist_check_entry
    solvation_dist_check[0] = solvation_dist_check_entry
    com_gtensor_dist_check[0] = com_gtensor_dist_check_entry
    connectedness_dist_check[0] = connectedness_dist_check_entry
    sticker_lifetime_dist_check[0] = sticker_lifetime_dist_check_entry
    concave_hull_dist_check[0] = concave_hull_dist_check_entry
    num_stickers[0] = sticker_array_entry_length
    fractal_ana_samples[0] = fractal_ana_samples_entry
    fractal_ana_radius[0] = fractal_ana_radius_entry
    fractal_ana_max_distance[0] = fractal_ana_max_distance_entry
    concave_hull_z_thickness[0] = concave_hull_z_thickness_entry
    concave_hull_z_requirement[0] = concave_hull_z_requirement_entry
    num_distinct_chains[0] = num_distinct_chains_entry

    for i in range(2 * RE_DIST_ARRAY_SIZE):
        re_dist_array_solo[i] = 0
        re_dist_array_clus[i] = 0

    for i in range(6 * GYRATION_TENSOR_DIST_ARRAY_SIZE):
        gtensor_dist_array[i] = 0

    for i in range(FRACTAL_MAX * FRACTAL_ANA_DIST_ARRAY_SIZE):
        fractal_ana_dist_array[i] = 0
        
    for i in range(CROSSLINK_MAX):
        crosslink_ana_array[i] = 0
    
    for i in range(CROSSLINK_COUNT_MAX):
        crosslink_count_array[i] = 0
        
    for i in range(SOLVATION_MAX * SOLVATION_ANA_DIST_ARRAY_SIZE):
        solvation_array[i] = 0

    for i in range(3 * DISPLACEMENT_ANA_DIST_ARRAY_SIZE):
        displacement_array[i] = 0

    for i in range(STICKER_LIFETIME_ARRAY_SIZE):
        sticker_lifetime_dist_array[i] = 0

    nu_file_size[0] = nu_size
    nu_file = &nu_file_list[0, 0]

    rg_full_file_size[0] = rg_full_size
    rg_full_file = &rg_full_file_list[0, 0]

    re_dist_file_size[0] = re_file_size
    re_file_clus = &re_dist_file_list[0, 0]
    re_file_solo = &re_dist_file_list[1, 0]

    gtensor_dist_file_size[0] = gtensor_file_size
    gtensor_file = &gtensor_dist_file_list[0, 0]    

    fractal_ana_dist_file_size[0] = fractal_ana_file_size
    fractal_ana_file = &fractal_ana_dist_file_list[0, 0]    

    rg_COM_dist_file_size[0] = rg_COM_file_size
    rg_COM_file = &rg_COM_file_list[0, 0]
    
    COM_bin_count_dist_file_size[0] = COM_bin_count_file_size
    COM_bin_count_file = &COM_bin_count_file_list[0, 0]

    COM_bead_dist_file_size[0] = COM_bead_file_size
    COM_bead_file = &COM_bead_file_list[0, 0]

    COM_chain_dist_file_size[0] = COM_chain_file_size
    COM_chain_file = &COM_chain_file_list[0, 0]

    COM_sticker_dist_file_size[0] = COM_sticker_file_size
    COM_sticker_file = &COM_sticker_file_list[0, 0]

    COM_Re_dist_file_size[0] = COM_Re_file_size
    COM_Re_file = &COM_Re_file_list[0, 0]

    COM_angle_dist_file_size[0] = COM_angle_file_size
    COM_angle_file = &COM_angle_file_list[0, 0]

    COM_dot_prod_dist_file_size[0] = COM_dot_prod_file_size
    COM_dot_prod_file = &COM_dot_prod_file_list[0, 0]

    COM_solvation_dist_file_size[0] = COM_solvation_file_size
    COM_solvation_file = &COM_solvation_file_list[0, 0]

    RDF_dist_file_size[0] = RDF_file_size
    RDF_file = &RDF_file_list[0, 0]

    mol_COM_dist_file_size[0] = mol_COM_file_size
    mol_COM_file = &mol_COM_file_list[0, 0]

    mol_COM_dist_file_size_2[0] = mol_COM_file_size_2
    mol_COM_file_2 = &mol_COM_file_list_2[0, 0]

    mol_COM_dist_file_size_3[0] = mol_COM_file_size_3
    mol_COM_file_3 = &mol_COM_file_list_3[0, 0]

    displacement_dist_file_size[0] = displacement_file_size
    displacement_file = &displacement_file_list[0, 0]

    sticker_cluster_dist_file_size[0] = sticker_cluster_file_size
    sticker_cluster_file = &sticker_cluster_file_list[0, 0]

    crosslink_dist_file_size[0] = crosslink_file_size
    crosslink_file = &crosslink_file_list[0, 0]
    
    cluster_dist_file_size[0] = cluster_file_size
    cluster_file = &cluster_file_list[0, 0]    

    solvation_dist_file_size[0] = solvation_file_size
    solvation_file = &solvation_file_list[0, 0]    

    COM_gtensor_dist_file_size[0] = COM_gtensor_file_size
    COM_gtensor_file = &COM_gtensor_file_list[0, 0]
    
    connectedness_dist_file_size[0] = connectedness_file_size
    connectedness_file = &connectedness_file_list[0, 0]

    connectedness_binned_dist_file_size[0] = connectedness_binned_file_size
    connectedness_binned_file = &connectedness_binned_file_list[0, 0]

    connectedness_full_dist_file_size[0] = connectedness_full_file_size
    connectedness_full_file = &connectedness_full_file_list[0, 0]

    sticker_lifetime_dist_file_size[0] = sticker_lifetime_file_size
    sticker_lifetime_file = &sticker_lifetime_file_list[0, 0]

    concave_hull_dist_file_size[0] = concave_hull_file_size
    concave_hull_file = &concave_hull_file_list[0, 0]

    sticker_array = <int *> calloc(num_stickers[0], sizeof(int))
    for i in range(num_stickers[0]):
        sticker_array[i] = sticker_array_entry[i]

    num_chains_array = <int *> calloc(num_distinct_chains[0], sizeof(int))
    for i in range(num_distinct_chains[0]):
        num_chains_array[i] = num_chains_array_entry[i]
        total_num_chains[0] += num_chains_array[i]
        
    chain_id_to_distinct_chain_array = <int *> calloc(total_num_chains[0], sizeof(int))
    j = 0
    k = 0
    for i in range(total_num_chains[0]):
        if k < num_chains_array[j]:
            chain_id_to_distinct_chain_array[i] = j
            k += 1
        else:
            j += 1
            chain_id_to_distinct_chain_array[i] = j
            k = 0

    connectedness_chains[0] = int(total_num_chains[0] / 20 / num_distinct_chains[0])
        
    chain_length_array = <int *> calloc(num_distinct_chains[0], sizeof(int))
    chain_id_to_first_bead_array = <int *> calloc(total_num_chains[0], sizeof(int))
    count_clus_array = <int *> calloc(num_distinct_chains[0], sizeof(int))
    count_solo_array = <int *> calloc(num_distinct_chains[0], sizeof(int))

    cur_file = &topo_file_list[0, 0]
    fp = fopen(cur_file[:topo_file_size], "r")
    while fgets(strLine, 200, fp) != NULL:

        if (<bytes> strLine).decode('utf-8').endswith('atoms\n'):
            num_beads[0] = int((<bytes> strLine).decode('utf-8').split()[0])
            frame_array = <int *> calloc(FRAME_ARRAY_MAX * num_beads[0], sizeof(int))
            topo_array = <int *> calloc(TOPO_ARRAY_MAX * num_beads[0], sizeof(int))
            bead_sticker_index_array = <int *> malloc(num_beads[0] * sizeof(int))
            if not frame_array or not topo_array:
                raise MemoryError()

        if (<bytes> strLine).decode('utf-8').endswith('atom types\n'):
            bead_types[0] = int((<bytes> strLine).decode('utf-8').split()[0])

        if (<bytes> strLine).decode('utf-8').endswith('xlo xhi\n'):
            box_size[0] = int((<bytes> strLine).decode('utf-8').split()[1])
            lattice_array = <int *> malloc(box_size[0] * box_size[0] * box_size[0] * sizeof(int))            

        if (<bytes> strLine).decode('utf-8').startswith('Atoms'):
            topo_check = 1
            continue
        
        if topo_check == 1 and (<bytes> strLine).decode('utf-8') != '\n':
            sscanf(strLine, "%d %d %d", &bead_count, &chain_id, &bead_type)
            topo_array[bead_count * TOPO_ARRAY_MAX + TOPO_ARRAY_CHAIN] = chain_id
            topo_array[bead_count * TOPO_ARRAY_MAX + TOPO_ARRAY_BEAD] = bead_type
            if bead_type >= bead_types[0]:
                bead_types[0] = bead_type + 1
            for i in range(num_stickers[0]):
                if bead_type == sticker_array[i]:
                    bead_sticker_index_array[bead_count] = total_num_stickers[0]
                    total_num_stickers[0] += 1
                    break
            else:
                bead_sticker_index_array[bead_count] = -1
            if bead_count == num_beads[0] - 1:
                break
    
    for i in range(num_beads[0]):
        chain_id = topo_array[i * TOPO_ARRAY_MAX + TOPO_ARRAY_CHAIN]
        chain_type = chain_id_to_distinct_chain_array[chain_id]
        if chain_length_array[chain_type] == 0:
            cur_chain_id = chain_id
        if chain_id == cur_chain_id:
            chain_length_array[chain_type] += 1
        if chain_id != last_chain_id:
            chain_id_to_first_bead_array[chain_id] = i
        last_chain_id = chain_id
    chain_length_fb_array = <int *> calloc(num_distinct_chains[0], sizeof(int))
    for i in range(num_distinct_chains[0]):
        chain_length_fb_array[i] = chain_lengths_sum[0]
        chain_lengths_sum[0] += chain_length_array[i]
    
    skip_frames = int((time_cutoff - time_start) / time_inc)
    tot_frames = int((time_end - time_start) / time_inc) + 1
    ana_frames[0] = tot_frames - skip_frames

    if nu_check[0] == 1:
        scaling_length_array = <int *> calloc(num_distinct_chains[0], sizeof(int))
        scaling_length_fb_array = <int *> calloc(num_distinct_chains[0], sizeof(int))
        for i in range(num_distinct_chains[0]):
            scaling_length_array[i] = int((chain_length_array[i] * chain_length_array[i] - chain_length_array[i]) / 2)
            scaling_length_fb_array[i] = total_scaling_length[0]
            total_scaling_length[0] += scaling_length_array[i]
        scaling_array_clus = <long double *> calloc(total_scaling_length[0], sizeof(long double))
        scaling_array_solo = <long double *> calloc(total_scaling_length[0], sizeof(long double))
        nu_clus_array = <long double *> calloc(num_distinct_chains[0], sizeof(long double))
        nu_clus_err_array = <long double *> calloc(num_distinct_chains[0], sizeof(long double))
        nu_solo_array = <long double *> calloc(num_distinct_chains[0], sizeof(long double))
        nu_solo_err_array = <long double *> calloc(num_distinct_chains[0], sizeof(long double))
        rg_full_array = <long double *> calloc(num_distinct_chains[0] * RG_FULL_MAX, sizeof(long double))
        rg_full_count_array = <long *> calloc(num_distinct_chains[0] * RG_FULL_MAX, sizeof(long))
                              
    if com_dist_check[0] == 1:
        COM_Rg_array = <long double *> calloc(4 * box_size[0] * num_distinct_chains[0], sizeof(long double))
        COM_bin_count_array = <long double *> calloc(4 * box_size[0] * num_distinct_chains[0], sizeof(long double))
        COM_bead_array = <long *> calloc(4 * box_size[0] * bead_types[0], sizeof(long))
        COM_chain_array = <long *> calloc(4 * 2 * box_size[0] * num_distinct_chains[0], sizeof(long))
        COM_count_array = <long *> calloc(4 * box_size[0] * num_distinct_chains[0], sizeof(long))
        COM_sticker_array = <long *> calloc(4 * 4 * box_size[0], sizeof(long))
        COM_sticker_count_array = <long *> calloc(4 * box_size[0], sizeof(long))
        COM_Re_array = <long double *> calloc(4 * box_size[0] * num_distinct_chains[0], sizeof(long double))
        COM_Re_count_array = <long *> calloc(4 * box_size[0] * num_distinct_chains[0], sizeof(long))
        COM_angle_array = <long double *> calloc(4 * box_size[0] * num_distinct_chains[0], sizeof(long double))
        COM_angle_count_array = <long *> calloc(4 * box_size[0] * num_distinct_chains[0], sizeof(long))
        COM_dot_prod_array = <long double *> calloc(4 * box_size[0] * chain_lengths_sum[0], sizeof(long double))
        COM_dot_prod_count_array = <long *> calloc(4 * box_size[0] * chain_lengths_sum[0], sizeof(long)) 
        COM_solvation_array = <long double *> calloc(4 * box_size[0] * num_distinct_chains[0] * COM_SOLVATION_MAX, sizeof(long double))
        COM_solvation_count_array = <long *> calloc(4 * box_size[0] * num_distinct_chains[0], sizeof(long))
        
    if RDF_dist_check[0] == 1:
        RDF_array = <long long *> calloc(4 * box_size[0], sizeof(long long))
        
    if mol_COM_dist_check[0] == 1:
        mol_COM_array = <long double *> calloc(4 * total_num_chains[0], sizeof(long double))
        mol_COM_array_old = <long double *> calloc(4 * total_num_chains[0], sizeof(long double))
        mol_COM_binned_array = <long double *> calloc(4 * box_size[0] * num_distinct_chains[0], sizeof(long double))
        mol_COM_binned_count_array = <long *> calloc(4 * box_size[0] * num_distinct_chains[0], sizeof(long))
        
    if sticker_cluster_dist_check[0] == 1:
        sticker_cluster_array = <int *> calloc(num_beads[0], sizeof(int))
    
    if com_gtensor_dist_check[0] == 1:
        COM_gtensor_array = <long double *> calloc(2 * 4 * box_size[0] * num_distinct_chains[0], sizeof(long double))
        COM_gtensor_count_array = <long *> calloc(4 * box_size[0] * num_distinct_chains[0], sizeof(long))
    
    if connectedness_dist_check[0] == 1:
        connectedness_array = <long double *> calloc(CONNECTEDNESS_ANA_DIST_ARRAY_SIZE * num_distinct_chains[0] * connectedness_chains[0] * CONNECTEDNESS_MAX, sizeof(long double))
        connectedness_binned_array = <long double *> calloc(BETWEENNESS_BINS * num_distinct_chains[0], sizeof(long double))
        connectedness_binned_count_array = <long *> calloc(BETWEENNESS_BINS * num_distinct_chains[0], sizeof(long))
        connectedness_full_array = <long double *> calloc(CONNECTEDNESS_FULL_ARRAY_SIZE * total_num_chains[0] * 3, sizeof(long double))
        connectedness_betweenness_array = <long double *> calloc(total_num_chains[0], sizeof(long double))
        connectedness_betweenness_array_old = <long double *> calloc(total_num_chains[0], sizeof(long double))
        connectedness_mol_COM_array = <long double *> calloc(3 * total_num_chains[0], sizeof(long double))
        connectedness_mol_COM_array_old = <long double *> calloc(3 * total_num_chains[0], sizeof(long double))
        connectedness_sticker_array = <int *> calloc(total_num_chains[0], sizeof(int))
        connectedness_sticker_array_old = <int *> calloc(total_num_chains[0], sizeof(int))

    if crosslink_dist_check[0] == 1:
        crosslink_ana_chain_array = <long double *> calloc(num_distinct_chains[0] * num_distinct_chains[0], sizeof(long double))
        crosslink_ana_full_array = <long double *> calloc(CROSSLINK_FULL_MAX * bead_types[0] * (bead_types[0] + 1), sizeof(long double))
        crosslink_ana_full_count_array = <long *> calloc(CROSSLINK_FULL_COUNT_MAX * bead_types[0], sizeof(long))

    if sticker_lifetime_dist_check[0] == 1:
        sticker_lifetime_array = <int *> calloc(<int> ((total_num_stickers[0] * total_num_stickers[0] - total_num_stickers[0]) / 2), sizeof(int))
    
    if concave_hull_dist_check[0] == 1:
        concave_hull_array = <int *> calloc(box_size[0] * box_size[0] * ana_frames[0], sizeof(int))
        concave_hull_com_array = <long double *> calloc(ana_frames[0] * 2, sizeof(long double))
        
    cluster_size_dilute_array = <int *> calloc(total_num_chains[0], sizeof(int))
    cluster_size_dense_array = <int *> calloc(total_num_chains[0], sizeof(int))
    
    for file_num in range(num_files):

        cur_file = &trj_file_list[file_num, 0]
        fp = fopen(cur_file[:trj_file_size], "r")
        
        if binary_check == 1:
            
            fseek(fp, sizeof(int) * skip_frames * (1 + 4 * num_beads[0]), SEEK_SET)
            cur_frame = skip_frames
            
            while cur_frame < tot_frames:
            
                fseek(fp, sizeof(int), SEEK_CUR)
                fread(frame_array, sizeof(int), FRAME_ARRAY_MAX * num_beads[0], fp)
                Perform_Analysis(frame_array, topo_array, lattice_array, scaling_array_clus,
                                 scaling_array_solo, scaling_length_array,
                                 scaling_length_fb_array, chain_length_fb_array, rg_full_array,
                                 rg_full_count_array, COM_Rg_array, COM_bin_count_array,
                                 COM_bead_array, COM_chain_array, COM_count_array,
                                 COM_sticker_array, COM_sticker_count_array, COM_Re_array,
                                 COM_Re_count_array, COM_angle_array,
                                 COM_angle_count_array, COM_dot_prod_array, COM_dot_prod_count_array,
                                 COM_solvation_array, COM_solvation_count_array, 
                                 COM_gtensor_array, COM_gtensor_count_array, RDF_array,
                                 mol_COM_array, mol_COM_array_old, mol_COM_binned_array,
                                 mol_COM_binned_count_array, sticker_cluster_array,
                                 connectedness_array, connectedness_binned_array,
                                 connectedness_binned_count_array, connectedness_full_array,
                                 connectedness_betweenness_array,
                                 connectedness_betweenness_array_old,
                                 connectedness_mol_COM_array,
                                 connectedness_mol_COM_array_old, connectedness_sticker_array,
                                 connectedness_sticker_array_old, sticker_lifetime_array,
                                 cluster_size_dilute_array,
                                 cluster_size_dense_array, sticker_array, bead_sticker_index_array,
                                 crosslink_ana_chain_array,
                                 crosslink_ana_full_array, crosslink_ana_full_count_array,
                                 concave_hull_array, concave_hull_com_array,
                                 num_chains_array, chain_id_to_distinct_chain_array,
                                 chain_id_to_first_bead_array, chain_length_array,
                                 count_clus_array, count_solo_array,
                                 re_file_clus, re_file_solo, gtensor_file, fractal_ana_file,
                                 mol_COM_file, mol_COM_file_2, displacement_file,
                                 solvation_file, connectedness_file, connectedness_full_file,
                                 sticker_lifetime_file)
                cur_frame += 1
        
        if binary_check == 0:

            line_index = 0
            while fgets(strLine, 200, fp) != NULL:
                
                if line_index < (skip_frames * (9 + num_beads[0])):
                    line_index += 1
                    continue
                
                elif (<bytes> strLine).decode('utf-8') == 'ITEM: ATOMS id type mol x y z bP\n':
                    sim_check = 1
                
                elif sim_check == 1:
                    sscanf(strLine, "%d %d %d %d %d %d %d", &bead_count, &bead_type, &chain_id, &x_pos, &y_pos, &z_pos, &bP)
                    frame_array[bead_count * FRAME_ARRAY_MAX + FRAME_ARRAY_X] = x_pos
                    frame_array[bead_count * FRAME_ARRAY_MAX + FRAME_ARRAY_Y] = y_pos
                    frame_array[bead_count * FRAME_ARRAY_MAX + FRAME_ARRAY_Z] = z_pos
                    frame_array[bead_count * FRAME_ARRAY_MAX + FRAME_ARRAY_BP] = bP
                    if bead_count == num_beads[0] - 1:
                        Perform_Analysis(frame_array, topo_array, lattice_array, scaling_array_clus,
                                         scaling_array_solo, scaling_length_array,
                                         scaling_length_fb_array, chain_length_fb_array, rg_full_array,
                                         rg_full_count_array, COM_Rg_array, COM_bin_count_array,
                                         COM_bead_array, COM_chain_array, COM_count_array,
                                         COM_sticker_array, COM_sticker_count_array, COM_Re_array,
                                         COM_Re_count_array, COM_angle_array,
                                         COM_angle_count_array, COM_dot_prod_array, COM_dot_prod_count_array,
                                         COM_solvation_array, COM_solvation_count_array, 
                                         COM_gtensor_array, COM_gtensor_count_array, RDF_array,
                                         mol_COM_array, mol_COM_array_old, mol_COM_binned_array,
                                         mol_COM_binned_count_array, sticker_cluster_array,
                                         connectedness_array, connectedness_binned_array,
                                         connectedness_binned_count_array, connectedness_full_array,
                                         connectedness_betweenness_array,
                                         connectedness_betweenness_array_old,
                                         connectedness_mol_COM_array,
                                         connectedness_mol_COM_array_old, connectedness_sticker_array,
                                         connectedness_sticker_array_old, sticker_lifetime_array,
                                         cluster_size_dilute_array,
                                         cluster_size_dense_array, sticker_array, bead_sticker_index_array,
                                         crosslink_ana_chain_array,
                                         crosslink_ana_full_array, crosslink_ana_full_count_array,
                                         concave_hull_array, concave_hull_com_array,
                                         num_chains_array, chain_id_to_distinct_chain_array,
                                         chain_id_to_first_bead_array, chain_length_array,
                                         count_clus_array, count_solo_array,
                                         re_file_clus, re_file_solo, gtensor_file, fractal_ana_file,
                                         mol_COM_file, mol_COM_file_2, displacement_file,
                                         solvation_file, connectedness_file, connectedness_full_file,
                                         sticker_lifetime_file)
                        sim_check = 0
                
                else:
                    continue
            
        fclose(fp)
    
    if nu_check[0] == 1:
        
        for i in range(num_distinct_chains[0]):

            if count_clus_array[i] != 0:
                for j in range(scaling_length_fb_array[i], scaling_length_fb_array[i] + scaling_length_array[i]):
                    scaling_array_clus[j] /= count_clus_array[i]
                Nu_Calculator(scaling_array_clus, i, scaling_length_fb_array,
                              scaling_length_array, chain_length_array, 0, 25)
                nu_clus_array[i] = nu_mean[0]
                nu_clus_err_array[i] = nu_err[0]
            
            if count_solo_array[i] != 0:
                for j in range(scaling_length_fb_array[i], scaling_length_fb_array[i] + scaling_length_array[i]):
                    scaling_array_solo[j] /= count_solo_array[i]
                Nu_Calculator(scaling_array_solo, i, scaling_length_fb_array,
                              scaling_length_array, chain_length_array, 0, 25)
                nu_solo_array[i] = nu_mean[0]
                nu_solo_err_array[i] = nu_err[0]
         
        Nu_Writer(nu_file, nu_clus_array, nu_clus_err_array, nu_solo_array, nu_solo_err_array)
        Rg_Full_Writer(rg_full_file, rg_full_array, rg_full_count_array)
        
        free(scaling_array_clus)
        free(scaling_array_solo)
        free(scaling_length_array)
        free(scaling_length_fb_array)
        free(nu_clus_array)
        free(nu_clus_err_array)
        free(nu_solo_array)
        free(nu_solo_err_array)
        free(rg_full_array)
        free(rg_full_count_array)

    if re_dist_check[0] == 1:
         
        if re_count_clus[0] > 0:
            Re_Dist_Writer_Clus(re_file_clus)
     
        if re_count_solo[0] > 0:
            Re_Dist_Writer_Solo(re_file_solo)

    if gtensor_dist_check[0] == 1:
        if gtensor_count[0] > 0:
            Gtensor_Dist_Writer(gtensor_file)

    if fractal_ana_dist_check[0] == 1:
        if fractal_ana_count[0] > 0:
            Fractal_Ana_Dist_Writer(fractal_ana_file)

    if com_dist_check[0] == 1:
 
        Rg_COM_Writer(rg_COM_file, COM_Rg_array, COM_count_array)
        COM_Bin_Count_Writer(COM_bin_count_file, COM_bin_count_array, COM_count_array)
        COM_Bead_Writer(COM_bead_file, COM_bead_array)
        COM_Chain_Writer(COM_chain_file, COM_chain_array)
        COM_Sticker_Writer(COM_sticker_file, COM_sticker_array, COM_sticker_count_array)
        COM_Re_Writer(COM_Re_file, COM_Re_array, COM_Re_count_array)
        COM_Angle_Writer(COM_angle_file, COM_angle_array, COM_angle_count_array)
        #=======================================================================
        # COM_Dot_Prod_Writer(COM_dot_prod_file, COM_dot_prod_array, COM_dot_prod_count_array,
        #                     chain_length_fb_array, chain_length_array)
        # COM_Solvation_Writer(COM_solvation_file, COM_solvation_array, COM_solvation_count_array)
        #=======================================================================
        free(COM_Rg_array)
        free(COM_bin_count_array)
        free(COM_bead_array)
        free(COM_chain_array)
        free(COM_count_array)
        free(COM_sticker_array)
        free(COM_sticker_count_array)
        free(COM_Re_array)
        free(COM_Re_count_array)
        free(COM_angle_array)
        free(COM_angle_count_array)
        free(COM_dot_prod_array)
        free(COM_dot_prod_count_array)
        free(COM_solvation_array)
        free(COM_solvation_count_array)

    if RDF_dist_check[0] == 1:
         
        RDF_Writer(RDF_file, RDF_array)
        free(RDF_array)

    if mol_COM_dist_check[0] == 1:
         
        mol_COM_Binned_Writer(mol_COM_file_3, mol_COM_binned_array, mol_COM_binned_count_array)
        Displacement_Writer(displacement_file)
        free(mol_COM_array)
        free(mol_COM_array_old)
        free(mol_COM_binned_array)
        free(mol_COM_binned_count_array)

    if sticker_cluster_dist_check[0] == 1:
         
        Sticker_Cluster_Writer(sticker_cluster_file, sticker_cluster_array)
        free(sticker_cluster_array)
    
    if crosslink_dist_check[0] == 1:
        
        Crosslink_Ana_Writer(crosslink_file, crosslink_ana_chain_array, crosslink_ana_full_array, crosslink_ana_full_count_array)
        free(crosslink_ana_chain_array)
        free(crosslink_ana_full_array)
        free(crosslink_ana_full_count_array)

    if solvation_dist_check[0] == 1:
         
        Solvation_Writer(solvation_file)

    if com_gtensor_dist_check[0] == 1:
         
        COM_GTensor_Writer(COM_gtensor_file, COM_gtensor_array, COM_gtensor_count_array)
        free(COM_gtensor_array)
        free(COM_gtensor_count_array)

    if connectedness_dist_check[0] == 1:
        
        Connectedness_Writer(connectedness_file, connectedness_array)
        Connectedness_Binned_Writer(connectedness_binned_file, connectedness_binned_array,
                                    connectedness_binned_count_array)
        Connectedness_Full_Writer(connectedness_full_file, connectedness_full_array)
        free(connectedness_array)
        free(connectedness_binned_array)
        free(connectedness_binned_count_array)
        free(connectedness_full_array)
        free(connectedness_betweenness_array)
        free(connectedness_betweenness_array_old)
        free(connectedness_mol_COM_array)
        free(connectedness_mol_COM_array_old)
        free(connectedness_sticker_array)
        free(connectedness_sticker_array_old)
    
    if sticker_lifetime_dist_check[0] == 1:
        
        Sticker_Lifetime_Writer(sticker_lifetime_file, sticker_lifetime_array, 1)
        free(sticker_lifetime_array)
    
    if concave_hull_dist_check[0] == 1:
        
        Concave_Hull_Writer(concave_hull_file, concave_hull_array, concave_hull_com_array)
        free(concave_hull_array)
        free(concave_hull_com_array)
        
    Cluster_Writer(cluster_file, cluster_size_dilute_array, cluster_size_dense_array)
    free(cluster_size_dilute_array)
    free(cluster_size_dense_array)
    
    free(frame_array)
    free(topo_array)
    free(lattice_array)
    free(num_chains_array)
    free(sticker_array)
    free(chain_length_array)
    free(chain_length_fb_array)
    free(chain_id_to_distinct_chain_array)
    free(chain_id_to_first_bead_array)
    free(count_clus_array)
    free(count_solo_array)

    return

