import os
import yaml
import argparse
import pandas as pd
import SimpleITK as sitk
import torch
import monai
from tqdm import tqdm

from segmentation_pl import UNetrPL, SwinUNETRPL, DYNUNETPL

def normalize_torchio(tensor:torch.Tensor)->torch.Tensor:
	m = tensor.mean()
	s = tensor.std(unbiased=True)
	return (tensor - m) / s

root_dir = "/home/ccig/anacarolina/procancer-i_segment"
parquet_path = '/home/jose_almeida/projects/pcai/data/series.parquet'
df = pd.read_parquet(parquet_path)
model_path = os.path.join(root_dir,'models/swin_unetr_fold_focal0_last.ckpt')
config_file = os.path.join(root_dir,'config/swin_unetr.yaml')
uc2_ids = list(dict.fromkeys(list(df.loc[(df['use_case'].str.contains('2')) | (df['use_case'].str.contains('5')) | (df['use_case'].str.contains('7')), 'patient_id'])))
print('Number of patients in UC2, UC5 and UC7: ', len(uc2_ids))

img_path = "/home/jose_almeida/data/ProCAncer-I"

size = [320,320,32]
t_input = monai.transforms.Compose(
	[monai.transforms.LoadImaged("image",ensure_channel_first=True,
				     image_only=True)])
t = monai.transforms.Compose(
    [monai.transforms.Orientationd("image","RAS"),
     monai.transforms.CenterSpatialCropd("image",size),
     monai.transforms.SpatialPadd("image",size)])
inverted_transforms = monai.transforms.Invertd("image",t)

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--force",action="store_true",
			    help="Predicts mask even if it already exists",
			    default=False)
	parser.add_argument("--tta",action="store_true",
			    help="Applies test-time augmentation (flips)",
			    default=False)
	parser.add_argument("--i",action="store",type=int,
			    help="Predicts only until the ith line of the parquet file",
			    default=None)
	parser.add_argument("--dev",action="store",type=str,
			    help="Device for model",
			    default="cuda")
	args = parser.parse_args()

	target = {}

	# load model here (no need to load it every time)
	with open(config_file,'r') as o:
		network_config = yaml.safe_load(o)

	# creates the model
	#  model must match the model in the path
	unet = SwinUNETRPL(
		image_key="image",label_key="label",
		**network_config)
	# model in eval mode
	unet.eval()
		
	# laod the trained weights
	state_dict = torch.load(model_path,map_location=args.dev)['state_dict']
	unet.load_state_dict(state_dict)
	unet = unet.to(args.dev)

	if args.i is None:
		i = len(uc2_ids)
	else:
		i = args.i
	pbar = tqdm(uc2_ids[0:i], desc='description')
	for p_id in pbar:
		folder1 = p_id[4]
		folder2 = p_id[5]
		u_id = df.loc[df['patient_id']==p_id, 'radiology_study_source_id'].iloc[0]
		p_path = os.path.join(img_path, folder1, folder2, p_id, u_id)
		pbar.set_description("ProCAncer-I ID={}".format(p_id))
		outdir = os.path.join(os.path.join(root_dir,'masks_AnaNuno'),
				      folder1, folder2, p_id)
		output_path = os.path.join(outdir, 'gland_mask_T2_AnaNuno.nii.gz')

		# skip prediction if output already exists
		if os.path.exists(output_path) == True and args.force == False:
			continue

		try:
			all_files = os.listdir(p_path)
		except FileNotFoundError:
			continue
		
		if 'image_T2.nii.gz' in all_files:
			T2_path = os.path.join(p_path, 'image_T2.nii.gz')
			
		if 'image_DWI.nii.gz' in all_files:
			DWI_path = os.path.join(p_path, 'image_DWI.nii.gz')

		if 'image_ADC.nii.gz' in all_files:
			ADC_path = os.path.join(p_path, 'image_ADC.nii.gz')

		
		with torch.no_grad():
			inputs = t_input({"image":T2_path})
			# torchio and monai normalize tensors differently (monai uses Bessel's correction,
			# torchio does not), so this is a shorthand way of normalizing the image
			inputs["image"] = normalize_torchio(inputs["image"])
			inputs = t(inputs)["image"].unsqueeze(0).to(args.dev)
			if args.tta == True:
				axes = ((2,),(3,),(4,))
				axes_inv = tuple([tuple([-x for x in d]) for d in axes])
				inputs = torch.cat([inputs,*[torch.flip(inputs,d) for d in axes]],0)
				output = unet.forward(inputs).detach()
				output = torch.stack(
					[output[0],
					 *[torch.flip(output[i],d) for i,d in enumerate(axes_inv)]],0)
				output = output.mean(axis=0,keepdim=True)
			else:
				output = unet.forward(inputs)
			output = output.argmax(dim=1, keepdim=True).detach().cpu()
			output = inverted_transforms({"image":output[0]})["image"]
			segmentation = sitk.Cast(sitk.GetImageFromArray(output.swapaxes(1,3)[0]),
						 sitk.sitkInt16)
		try:	
			os.makedirs(outdir,exist_ok=True)
		except FileExistsError:
			print('FileExistsError', i,  p_id)
		
		inputImage = sitk.ReadImage(T2_path)	
		segmentation.CopyInformation(inputImage)
		
		sitk.WriteImage(segmentation, output_path)

		g1 = df.loc[df['patient_id']==p_id,'gleason1'].iloc[0]
		g2 = df.loc[df['patient_id']==p_id,'gleason2'].iloc[0]
		target[p_id]=(g1, g2, g1+g2, g1+g2>6)
	
