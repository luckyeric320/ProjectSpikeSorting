import time

from django.shortcuts import render
from django.shortcuts import HttpResponse
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile, File
from django.conf import settings
import json,os
import SplcV3
import uuid
# Create your views here.
def index(request):
    return render(request, 'index.html')
def dl(request):
    filename = request.GET.get('file')
    file_pathname = os.path.join(os.path.join(settings.BASE_DIR,'results'), filename)

    with open(file_pathname, 'rb') as f:
        file = File(f)
        response = HttpResponse(file.chunks(),
                                content_type='APPLICATION/OCTET-STREAM')
        response['Content-Disposition'] = 'attachment; filename=' + filename
        response['Content-Length'] = os.path.getsize(file_pathname)

    return response
def up(request):
    img=request.FILES.get('file')
    print(img)
    data = {
        "name":img.name,
        "size":img.size
    }
    uuid_str = uuid.uuid4().hex
    tmp_file = os.path.join(settings.UPLOAD_URL, "HF_{}.npy".format(uuid_str))
    path1=default_storage.save(tmp_file,ContentFile(img.read()))
    file = '{}.zip'.format(int(time.time()))
    zipfilename=os.path.join(settings.BASE_DIR,"results/"+file)
    SplcV3.visualization(os.path.join(settings.BASE_DIR,"codes"),tmp_file,os.path.join(settings.BASE_DIR,'results'),zipfilename)
    print(file)
    return render(request, 'download.html',{'file': file})
    # return HttpResponse(json.dumps(data))





