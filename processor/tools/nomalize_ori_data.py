import shutil,os
import dicom

ds_folder = '/home/train/'
labelsuffix=['AC', 'LAD', 'LCX', 'RCA']

def validate(case, suit_size):
    min_size = min(suit_size)
    ending = 0
    forward = 5
    chaos_maintain = True
    for i in range(0, len(suit_size)):
        if ending >= 10:
            if suit_size[i] > min_size:
                for j in range(1, forward):
                    if i + j < len(suit_size) and suit_size[i + j] > min_size:
                        chaos_maintain = True
                    else:
                        ending = 0
                        chaos_maintain = False
                        break
                if chaos_maintain:
                    print("case %s -> chaos found on: %d" % (case, i))
                    return case
        if suit_size[i] == min_size:
            ending += 1
        else:
            ending = 0
            #     print( "no chaos found")
    return None


def chaos_checking(case, ds_folder):
    #     print("checking %s" % case)
    files = os.listdir(os.path.join(ds_folder, case))
    dcmfiles = filter(lambda x: x.endswith('dcm'), files)
    dcmfiles = sorted(dcmfiles, key=lambda x: int(x.split('_')[1].replace(".dcm", "")))
    suit_sizes = []
    #     print(dcmfiles)
    for i in range(0, len(dcmfiles)):
        suit_index = int(dcmfiles[i].split('_')[1].replace(".dcm", ""))
        s_size = 0
        for label in labelsuffix:
            f_path = os.path.join(ds_folder, case, "%s_%s.png" % (dcmfiles[i].replace(".dcm", ""), label))
            if os.path.exists(f_path):
                s_size += os.path.getsize(f_path)
        suit_sizes.append(s_size)
    # print(suit_sizes)
    return validate(case, suit_sizes)


def handle_case(case, ds_folder):
    files = os.listdir(os.path.join(ds_folder, case))
    files.sort()
    print("start handling case %s" % case)
    c_id = case
    dcmcount = 0
    jpgcount = 0
    for i in range(0, len(files)):
        file = files[i]
        if file.endswith('dcm'):
            filename_base = file.replace(".dcm", "")
            ds = dicom.read_file(os.path.join(ds_folder, case, file))
            dcm_index = ds.InstanceNumber
            print(filename_base)
            print("modify order from %d to %d" % (dcmcount, dcm_index))
            newname = os.path.join(ds_folder, case, "%d.dcm" % dcm_index)
            #             os.rename(os.path.join(ds_folder,case,file), newname)

            newjpgname = os.path.join(ds_folder, case, "%d.jpg" % dcm_index)
            if os.path.exists(os.path.join(ds_folder, case, "%s.jpg" % filename_base)):
                print("rename jpg: %s to %s" % (os.path.join(ds_folder, case, "%s.jpg" % filename_base), newjpgname))
            # os.rename(os.path.join(ds_folder, case,"%s.jpg" % filename_base), newjpgname)

            for label in labelsuffix:
                label_file = os.path.join(ds_folder, case, "%s_%s.png" % (filename_base, label))
                if os.path.exists(label_file):
                    new_label_file_name = os.path.join(ds_folder, case, "%d_%s.png" % (dcm_index, label))
                    print("rename label: %s to %s" % (label_file, new_label_file_name))
                    #                     os.rename(label_file, new_label_file_name)
            dcmcount += 1

    files = os.listdir(os.path.join(ds_folder, case))
    dcmfiles = filter(lambda x: x.endswith('dcm'), files)
    dcmfiles = sorted(dcmfiles, key=lambda x: int(x.split('.')[0]))
    for i in range(0, len(dcmfiles)):
        f = dcmfiles[i]
        filename_base = f.replace(".dcm", "")
        print("rename file %s to  %s_%04d.dcm" % (f, c_id, i))
        newname = os.path.join(ds_folder, case, "%s_%04d.dcm" % (c_id, i))
        #         os.rename(os.path.join(ds_folder,case,f), newname)

        newjpgname = os.path.join(ds_folder, case, "%s_%04d.jpg" % (c_id, i))
        if os.path.exists(os.path.join(ds_folder, case, "%s.jpg" % filename_base)):
            print(newjpgname)
        # os.rename(os.path.join(ds_folder, case,"%s.jpg" % filename_base), newjpgname)

        for label in labelsuffix:
            label_file = os.path.join(ds_folder, case, "%s_%s.png" % (filename_base, label))
            if os.path.exists(label_file):
                new_label_file_name = os.path.join(ds_folder, case, "%s_%04d_%s.png" % (c_id, i, label))
                print("rename label: %s to %s" % (label_file, new_label_file_name))
                #                 os.rename(label_file, new_label_file_name)
    print("case %s completed" % case)


# for case in os.listdir(ds_folder):
#     handle_case(case, ds_folder)
#     break
# chaos_checking("JF93075125", ds_folder)
# chaos_checking("PF39886757", ds_folder)
# chaos_checking("WL63864474", ds_folder)
chaos_list = []
for case in os.listdir(ds_folder):
    try:
        c_id = chaos_checking(case, ds_folder)
        if c_id is not None:
            chaos_list.append(c_id)
    except Exception as e:
        pass

print(chaos_list)