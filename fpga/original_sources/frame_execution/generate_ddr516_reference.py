"""Generate an isolated Ti60 2Gb/x16/400MHz IP 5.16 reference via Efinity API.

Run with the installed Efinity setup.bat environment. No board downloads.
"""
import argparse,json,os,sys,shutil
from pathlib import Path

sys.path.append(str(Path(os.environ['EFXIPM_HOME'])/'bin'))
from ipm_api_service.design import IPMDesignAPI

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
 root=a.root.resolve();assert root.is_relative_to(Path('E:/fpga/tmp')) and not root.exists()
 root.mkdir(parents=True)
 # The IP API obtains project metadata from a valid project XML. This is
 # generation only: the engboard pins/project are never compiled or downloaded.
 package=Path(os.environ['EFINITY_HOME'])/'ipm/ip/efx_ddr3_soft_controller'
 shutil.copy2(package/'fpga/Ti60F225_engboard/ti60f225_ddr3_controller.xml',root/'ti60f225_ddr3_controller.xml')
 api=IPMDesignAPI(device_name='Ti60F225',family_name='Titanium',project_path=root,is_verbose=True)
 name='ddr516_reference'
 created=api.create_ip(module_name=name,vendor='efinixinc.com',library='memory_controller',name='efx_ddr3_soft_controller')
 assert created is not None
 config={'data_width':'1','freq':'400','row':'14','col':'10','bank':'3','rank':'1',
  'simulation':"1'b0",'DDR3_DATA_RATE':'2','cas_latency':'6','DDR3_MODE':'2'}
 api.config_ip(module_name=name,configs=config)
 ok,validation,params=api.validate_ip(module_name=name)
 resolved={x.name:x.value for x in params}
 (root/'generation_request.json').write_text(json.dumps(dict(config=config,resolved=resolved,validation=str(validation)),indent=2))
 assert ok,validation
 for k,v in config.items():assert resolved[k]==v,(k,resolved[k],v)
 result=api.generate_ip(module_name=name)
 assert getattr(result,'name','')=='SUCCESS' or getattr(result,'value',None)==0,result
 print('GENERATED',root,flush=True)

if __name__=='__main__':main()
