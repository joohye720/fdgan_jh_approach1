#!/bin/bash
#!/joohye/bin/env

folder_list=("cam1" "cam2" "cam3" "cam4" "cam5" "cam6")
mkdir images

for i in $(seq 0 532)
do
    for j in $(seq 0 1 5)
    do
        i_modified=$(($i+1))
        i_modified=$(printf "%04g" $i_modified)
        folder_idx=$(($j))
        cam_folder=${folder_list[$folder_idx]}
        echo /$cam_folder/$i_modified
	n_imgs=$(ls -l ./$cam_folder/$i_modified | wc -l)
        n_imgs=$(($n_imgs-1))
        n_imgs=$(printf "%04d" $n_imgs)

        for ii in $(seq -f "%04g" 1 $n_imgs)
        do
		
		identity=$(printf "%08g" $i)
        	cam_idx=$(printf "%02d" $j)
        	img_idx=$(printf "%04g" $ii)
		echo ${identity}_${cam_idx}_${img_idx}.jpg
                echo ./images/${identity}_${cam_idx}_${img_idx}.jpg
                cp ./$cam_folder/$i_modified/$ii.jpg ./images/${identity}_${cam_idx}_${img_idx}.jpg
        done	

    done  
  echo $i
done

