import h5py
import numpy as np
import pandas as pd 
import scipy.stats as stats
import re
import yaml
import edt
import ast
import gc
import argparse

#also install openpyxl into conda env

#export totals and (mean, sem, n) over all neurons

#ONLY CONSIDER LV

D0 = '/data/projects/weilab/dataset/hydra/results/'
D1 = '/home/rothmr/projects/stitched/stitched/' #stitched box files source dir
sample_dir = '/home/rothmr/hydra/sample/'

cache = {}


#use for full names list
names_20 = ["KR4", "KR5", "KR6", "SHL55", "PN3", "LUX2", "SHL20", "KR11", "KR10", 
			"RGC2", "KM4", "NET12", "NET10", "NET11", "PN7", "SHL18", 
			"SHL24", "SHL28", "RGC7", "SHL17"]

#neuron ID and type info
neuron_info = {"KR4": [1, 5], "KR5": [2, 2], "KR6": [4, 2], "SHL55": [3, 5], "PN3": [10, 3], "LUX2": [11, 5], "SHL20": [17, 5], "KR11": [8, 5], "KR10": [7, 4], 
			"RGC2": [6, 5], "KM4": [9, 5], "NET12": [15, 5], "NET10": [16, 1], "NET11": [14, 1], "PN7": [12, 5], "SHL18": [18, 1], 
			"SHL24": [5, 1], "SHL28": [20, 1], "RGC7": [13, 1], "SHL17": [19, 4]}

def neuron_id_to_name(nid):
	reversed_dict = {v: k for k, v in neuron_dict.items()}
	if isinstance(nid, int):
		nid = [nid]
	return [reversed_dict[x] for x in nid]  


def read_txt_to_dict(name, which):
	results_dict = {}

	if(name=="sample"):
		file_path = f'{sample_dir}sample_outputs/sample_com_mapping.txt'

	else:
		file_path = f'/home/rothmr/hydra/meta/new_meta/{name}_{which}_com_mapping.txt'

	with open(file_path, 'r') as file:
		for line in file:
			key_string, value_string = line.split(": ")
			#value_string = re.sub(r'\b(lv_\d+)\b', r"'\1'", value_string) #change into string literals
			value_string = re.sub(r"(?<!['\"])(\blv_\d+\b|\bnew\b|\blv\b)(?!['\"])", r"'\1'", value_string)
			#value_string = re.sub(r'\b(sv_\d+)\b', r"'\1'", value_string) #same for sv
			value_string = re.sub(r"(?<!['\"])(\bsv_\d+\b|\bnew\b|\blv\b)(?!['\"])", r"'\1'", value_string)

			coords_str = key_string.replace("[", "").replace("]", "").strip()
			coords_str = re.sub(r'[\(\)\,]', '', coords_str) #strip parens
			#print("cleaned: ", coords_str)

			coords_list=coords_str.split()
			coords = [float(coord) for coord in coords_list]

			attributes = ast.literal_eval(value_string)
			results_dict[tuple(coords)]=list(attributes)

	return results_dict


#takes in segid->classification labeling file
def lv_labels_dict(file_path):
	result_dict = {}
	with open(file_path, 'r') as file:
		for line in file:
			line = line.strip()
			pairs = line.strip('()').split('),(')

			for pair in pairs:
				key, value = pair.split(':')
				result_dict[int(key)] = int(value)

	return result_dict


def load_data(name): #load into cache
	print("loading for: ", name)

	if(name=="sample"):
		file_path = f'{sample_dir}sample_data/7-13_mask.h5'
	else:
		file_path = f"{D1}neuron_{name}_box_30-32-32.h5"

	with h5py.File(file_path, 'r') as f: #stitched mask, contains everything w segids
		cache["box"] = np.array(f["main"]).astype(int)
	print("done loading neuron box")

	cache["lv_mapping"] = read_txt_to_dict(name, "lv")



#outputs the num of vesicles within and the total
#9 means just everything
def calculate_vesicles_within(name, mask_name, which=9): #use data from cache - after loading
	#note that all data in the mapping txt file is in nm
	mask = cache[mask_name]

	#lv
	lv_mapping = cache["lv_mapping"] #com-> attributes metadata

	if(name=="sample"):
		types_dict_path = f"{sample_dir}sample_data/7-13_lv_label.txt"

	else:
		types_dict_path = f"/home/rothmr/hydra/types/new_types/new_v0+v2/{name}_lv_label.txt"

	labels_dict = lv_labels_dict(types_dict_path)

	total_num = 0
	num_in_mask = 0
	
	for com, attributes in lv_mapping.items():
		#find subtype
		subtype = 0
		label = int(attributes[1][3:]) #just the number; could be overlap

		if(label in labels_dict.keys()):
			subtype = labels_dict[label]
		else: #if doesn't exist in the dict, set to 0 - we ignore unclassified vesicles
			subtype = 0
			

		##now, subtype should have the correct value
		if((which==9 and subtype!=0) or (subtype==which)): #if we are including everything, or if subtype matches up
			#then consider this vesicle
			total_num +=1

			voxels_com = [com[0], com[1]/4, com[2]/4] #downsample since mask in 30-32-32
			voxels_com = np.round(voxels_com).astype(int) #round for indexing in the mask
			if (mask[tuple(voxels_com)]!=0): #boolean
				num_in_mask += 1

	print("total num: ", total_num)
	print("num in region: ", num_in_mask)


	return total_num, num_in_mask


def expand_mask(original_mask, threshold_nm, res): #threshold in physical units, res in xyz
	expanded_mask = np.copy(original_mask)

	print("begin distance transform for expansion")
	dt = edt.edt(1 - original_mask, anisotropy=(res[2],res[1],res[0])) #needs to be in xyz by default for edt
	print("end distance transform for expansion")
	doubled_perimeter = dt <= threshold_nm #all in nm
	expanded_mask[doubled_perimeter] = 1

	expanded_mask_binary = (expanded_mask>=1).astype(int)
	return expanded_mask_binary

#takes in a list of masks, outputs a binary mask
def mask_intersection(mask_list): #assume same size files, assume list has size at least 1
	intersection = np.bitwise_and.reduce(mask_list)
	return intersection

#calculate the volume given a boolean mask & convert to nm^3 units
def calculate_volume_nm(mask, res):
	binary_mask = (mask>=1).astype(int)
	volume_voxels = np.sum(binary_mask)
	volume_nm = volume_voxels * (res[0]*res[1]*res[2])
	return volume_nm


def neuron_name_to_id(name): #for a list of names and returns list??
	if isinstance(name, str):
		name = [name]
	return [neuron_dict[x] for x in name]  

def read_yml(filename):
    with open(filename, 'r') as file:
        data = yaml.safe_load(file)
    return data


if __name__ == "__main__":

	neuron_dict = read_yml('/projects/weilab/dataset/hydra/mask_mip1/neuron_id.txt')
	names = names_20
	mask_res = [30, 32, 32]

	results = [] #for xlxs export

	all_percentages_in = []


	#####
	parser = argparse.ArgumentParser()
	parser.add_argument("--which_neurons", type=str, help="all or sample?") #enter as "all" or "sample"
	parser.add_argument("--target_segid", type=float, help="target segid") #enter as an integer, only if sample
	parser.add_argument("--lv_threshold", type=float, help="lv threshold") 
	parser.add_argument("--cv_threshold", type=float, help="cv threshold") 
	parser.add_argument("--dv_threshold", type=float, help="dv threshold") 
	parser.add_argument("--dvh_threshold", type=float, help="dvh threshold") 
	args = parser.parse_args()
	which_neurons = args.which_neurons
	target_segid = args.target_segid
	lv_threshold = args.lv_threshold
	cv_threshold = args.cv_threshold
	dv_threshold = args.dv_threshold
	dvh_threshold = args.dvh_threshold


	#ensure which_neurons is entered
	if(args.which_neurons is None):
		parser.error("error - must enter all or sample for --which_neurons")

	#ensure target segid and thresholds added if sample
	if(args.which_neurons=="sample" and target_segid is None):
		parser.error("--target_segid required if --which_neurons is sample")

	if(args.which_neurons=="sample" and lv_threshold is None):
		parser.error("--lv_threshold required")

	if(args.which_neurons=="sample" and cv_threshold is None):
		parser.error("--cv_threshold required")

	if(args.which_neurons=="sample" and dv_threshold is None):
		parser.error("--dv_threshold required")

	if(args.which_neurons=="sample" and dvh_threshold is None):
		parser.error("--dvh_threshold required")

    #####

	if(which_neurons=="sample"):
		name = "sample" 
		neuron_id = 62
		neuron_type = "n/a" #for the neuron typings, not relevant for sample
		load_data(name)
		print(f"loaded data for {name}")
		nid = target_segid
		neurons = cache["box"]

		#extract regions of interest from mask
		neurons = cache["box"]

		neuron_only = np.zeros(neurons.shape, dtype=neurons.dtype)
		neuron_only[neurons==nid] = 1 #binary mask

		other_neurons = np.zeros(neurons.shape, dtype = neurons.dtype)
		other_neurons[(neurons!=nid) & (neurons!=0)] = 1 #binary mask
		print("final size of other_neurons mask in voxels: ", np.sum(other_neurons)) #error checking


		print("LV")
		expanded_others = expand_mask(other_neurons, lv_threshold, mask_res)
		print("mask expanding done")
		cache["intersections"] = mask_intersection((neuron_only, expanded_others))
		intersection_volume = np.sum(cache["intersections"])*(mask_res[0]*mask_res[1]*mask_res2) #nm
		print("size of intersection region: ", intersection_volume)
		print("mask intersection done")
		total_num, nn_lv = calculate_vesicles_within(name, "intersections") #name of cache file
		num_far = total_num-nn_lv
		print("num outside of region: ", num_far)
		del expanded_others, cache["intersections"]
		gc.collect()
		#append to individual row of the results output
		results.append([name, neuron_id, neuron_type, "LV", total_num, nn_lv, num_far, lv_threshold])

		print(f"{name} LV done \n")
	


		#cv
		print("CV")
		expanded_others = expand_mask(other_neurons, cv_threshold, mask_res)
		print("mask expanding done")
		cache["intersections"] = mask_intersection((neuron_only, expanded_others))
		intersection_volume = np.sum(cache["intersections"])*(mask_res[0]*mask_res[1]*mask_res2) #nm
		print("size of intersection region: ", intersection_volume)
		print("mask intersection done")
		total_num, nn_lv = calculate_vesicles_within(name, "intersections", which=1) #name of cache file
		num_far = total_num-nn_lv
		print("num outside of region: ", num_far)
		del expanded_others, cache["intersections"]
		gc.collect()
		#append to individual row of the results output
		results.append([name, neuron_id, neuron_type, "CV", total_num, nn_lv, num_far, cv_threshold])
		print(f"{name} CV done \n")



		#dv
		print("DV")
		expanded_others = expand_mask(other_neurons, dv_threshold, mask_res)
		print("mask expanding done")
		cache["intersections"] = mask_intersection((neuron_only, expanded_others))
		intersection_volume = np.sum(cache["intersections"])*(mask_res[0]*mask_res[1]*mask_res2) #nm
		print("size of intersection region: ", intersection_volume)
		print("mask intersection done")
		total_num, nn_lv = calculate_vesicles_within(name, "intersections", which=2) #name of cache file
		num_far = total_num-nn_lv
		print("num outside of region: ", num_far)
		del expanded_others, cache["intersections"]
		gc.collect()
		#append to individual row of the results output
		results.append([name, neuron_id, neuron_type, "DV", total_num, nn_lv, num_far, dv_threshold])
		print(f"{name} DV done \n")



		#dvh
		print("DVH")
		expanded_others = expand_mask(other_neurons, dv_threshold, mask_res)
		print("mask expanding done")
		cache["intersections"] = mask_intersection((neuron_only, expanded_others))
		intersection_volume = np.sum(cache["intersections"])*(mask_res[0]*mask_res[1]*mask_res2) #nm
		print("size of intersection region: ", intersection_volume)
		print("mask intersection done")
		total_num, nn_lv = calculate_vesicles_within(name, "intersections", which=3) #name of cache file
		num_far = total_num-nn_lv
		print("num outside of region: ", num_far)
		del expanded_others, cache["intersections"]
		gc.collect()
		#append to individual row of the results output
		results.append([name, neuron_id, neuron_type, "DVH", total_num, nn_lv, num_far, dvh_threshold])

		print(f"{name} DVH done\n")


	#loop thru all neurons
	elif(which_neurons=="all"):
		names_list = ["SHL24", "NET11", "RGC7", "NET10", "SHL18", "SHL28", "KR5", "KR6", "PN3"]

		for name in names_list:
			#neuron id/type info
			neuron_id = neuron_info[name][0]
			neuron_type = neuron_info[name][1]


			load_data(name) #load neurons and mappings
			print(f"loaded data for {name}")
			nid = neuron_name_to_id(name)

			###

			#extract regions of interest from mask
			neurons = cache["box"]

			neuron_only = np.zeros(neurons.shape, dtype=neurons.dtype)
			neuron_only[neurons==nid] = 1 #binary mask

			other_neurons = np.zeros(neurons.shape, dtype = neurons.dtype)

			#restrict to only neurons w these names:
			included_neuron_types = [1,2,3]
			other_included_names = ["SHL29", "SHL53", "PN8", "SHL26", "SHL51", "KM1", "KM2", "SHL52"]
			total_to_include = [n for n in neuron_dict.keys() if ((n in other_included_names) or (n in neuron_info.keys() and neuron_info[n][1] in included_neuron_types))]
			included_ids = [neuron_dict[n] for n in total_to_include] #turns list of names into list of IDs
			print("included ids: ", included_ids) #overall
			print("total num of IDs: ", len(neuron_dict), "num of potentially included IDs: ", len(included_ids))

			#if doing for all overlaps, ignoring specific handshake info
			'''
			other_neurons[(neurons!=nid) & (neurons!=0) & np.isin(neurons, included_ids)] = 1 #binary mask
			print("final size of other_neurons mask in voxels: ", np.sum(other_neurons))

			removed_neurons = [] #tracking difference when restricting to 1,2,3,unrestricted neuron types.
			for segid in np.unique(neurons):
				if(segid not in included_ids):
					removed_neurons.append(segid)
			print("removed neurons: ", removed_neurons)
			if(removed_neurons==[0]):
				print("NO CHANGE for ", name)

			handshake_name = "all" #for dataframe uniformity
			'''

			for handshake_id in included_ids:

				handshake_name = neuron_id_to_name(handshake_id)[0]
				print("handshake name: ", handshake_name)
				#handshake_type = neuron_info[handshake_name][1] #might not be complete
				if handshake_name not in neuron_info:
					print("error - type of handshaking neuron is missing")
				

				other_neurons[(neurons!=nid) & (neurons!=0) & np.isin(neurons, handshake_id)] = 1 #just the one handshake
				
				#only append to dataframe if vesicles within is nonzero
				#expand by sufficient amt, check if any nonzero intersection with target
				print("check for nonzero intersection")
				expanded_others = expand_mask(other_neurons, 830, mask_res) #choose a threshold larger than all LV, CV, DV, DVH thresholds
				intersection = np.bitwise_and.reduce([neuron_only, expanded_others])
				volume = np.sum(intersection) #voxels

				print("voxel intersection volume of ", handshake_name, "with", name, "is", volume)
				if(volume > 0): 
					print("continuing\n")


					print("LV")
					expanded_others = expand_mask(other_neurons, lv_threshold, mask_res)
					print("mask expanding done")
					cache["intersections"] = mask_intersection((neuron_only, expanded_others))
					intersection_volume = np.sum(cache["intersections"])*(mask_res[0]*mask_res[1]*mask_res2) #nm
					print("size of intersection region: ", intersection_volume)
					print("mask intersection done")
					total_num, nn_lv = calculate_vesicles_within(name, "intersections") #name of cache file
					num_far = total_num-nn_lv
					print("num outside of region: ", num_far)
					del expanded_others, cache["intersections"]
					gc.collect()
					#append to individual row of the results output
					results.append([name, neuron_id, neuron_type, handshake_name, handshake_id, intersection_volume, "LV", total_num, nn_lv, num_far, lv_threshold])

					print(f"{name} LV done \n")
				


					#cv
					print("CV")
					expanded_others = expand_mask(other_neurons, cv_threshold, mask_res)
					print("mask expanding done")
					cache["intersections"] = mask_intersection((neuron_only, expanded_others))
					intersection_volume = np.sum(cache["intersections"])*(mask_res[0]*mask_res[1]*mask_res2) #nm
					print("size of intersection region: ", intersection_volume)
					print("mask intersection done")
					total_num, nn_lv = calculate_vesicles_within(name, "intersections", which=1) #name of cache file
					num_far = total_num-nn_lv
					print("num outside of region: ", num_far)
					del expanded_others, cache["intersections"]
					gc.collect()
					#append to individual row of the results output
					results.append([name, neuron_id, neuron_type, handshake_name, handshake_id, intersection_volume, "CV", total_num, nn_lv, num_far, cv_threshold])
					print(f"{name} CV done \n")



					#dv
					print("DV")
					expanded_others = expand_mask(other_neurons, dv_threshold, mask_res)
					print("mask expanding done")
					cache["intersections"] = mask_intersection((neuron_only, expanded_others))
					intersection_volume = np.sum(cache["intersections"])*(mask_res[0]*mask_res[1]*mask_res2) #nm
					print("size of intersection region: ", intersection_volume)
					print("mask intersection done")
					total_num, nn_lv = calculate_vesicles_within(name, "intersections", which=2) #name of cache file
					num_far = total_num-nn_lv
					print("num outside of region: ", num_far)
					del expanded_others, cache["intersections"]
					gc.collect()
					#append to individual row of the results output
					results.append([name, neuron_id, neuron_type, handshake_name, handshake_id, intersection_volume, "DV", total_num, nn_lv, num_far, dv_threshold])
					print(f"{name} DV done \n")


					#dvh
					print("DVH")
					expanded_others = expand_mask(other_neurons, dvh_threshold, mask_res)
					print("mask expanding done")
					cache["intersections"] = mask_intersection((neuron_only, expanded_others))
					intersection_volume = np.sum(cache["intersections"])*(mask_res[0]*mask_res[1]*mask_res2) #nm
					print("size of intersection region: ", intersection_volume)
					print("mask intersection done")
					total_num, nn_lv = calculate_vesicles_within(name, "intersections", which=3) #name of cache file
					num_far = total_num-nn_lv
					print("num outside of region: ", num_far)
					del expanded_others, cache["intersections"]
					gc.collect()
					#append to individual row of the results output
					results.append([name, neuron_id, neuron_type, handshake_name, handshake_id, intersection_volume, "DVH", total_num, nn_lv, num_far, dvh_threshold])

					print(f"{name} DVH done\n")



	df = pd.DataFrame(results, columns=["Neuron", "Neuron ID", "Neuron type", "Handshake name", "Handshake ID", "Intersection volume", "Vesicle type", "Total num", "Near neuron", "Not near neuron", "Nearness threshold (nm)"])
	if(which_neurons=="all"):
		export_path = "/home/rothmr/hydra/sheet_exports/near_handshake_counts.xlsx"
	elif(which_neurons=="sample"):
		export_path = f"{sample_dir}sample_outputs/sample_near_counts.xlsx"
	df.to_excel(export_path, index=False)
	print("export done")
	print("done")


	#pointcloud_near_counts.py --which_neurons "all" --lv_threshold 708.4388202 --cv_threshold 756.0435442 --dv_threshold 546.4010465 --dvh_threshold 827.825424

