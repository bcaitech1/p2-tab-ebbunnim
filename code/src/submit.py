import json
import os

import requests


def submit(user_key='',file_path=''):
    if not user_key:
        raise Exception("No user-key")
    url='http://ec2-13-124-161-225.ap-northeast-2.compute.amazonaws.com:8000/api/v1/competition/3/presigned_url/?description=&hyperparameters={%22training%22:{},%22inference%22:{}}'
    headers={
        'Authorization':user_key
    }
    res=requests.get(url,headers=headers)
    data=json.loads(res.text)
    submit_url=data['url']
    body = {
        'key':'app/Competitions/000003/Users/{}/Submissions/{}/output.csv'.format(str(data['submission']['user']).zfill(8),str(data['submission']['local_id']).zfill(4)),
        'x-amz-algorithm':data['fields']['x-amz-algorithm'],
        'x-amz-credential':data['fields']['x-amz-credential'],
        'x-amz-date':data['fields']['x-amz-date'],
        'policy':data['fields']['policy'],
        'x-amz-signature':data['fields']['x-amz-signature']
    }
    requests.post(url=submit_url, data=body, files={'file': open(file_path, 'rb')}) 

output_dir= '/opt/ml/code/output'
# user_key='{config 파일에 담아둠}'
submit(user_key, os.path.join(output_dir, 'output.csv'))
print("Done submit!")


