"""Equivalent O(1) host-side TMDS checker delay; no DUT or timing changes."""
DECL = 'reg [23:0] rgb_delay[0:58];reg de_delay[0:58];reg [15:0] frame_delay[0:58];reg [15:0] x_delay[0:58],y_delay[0:58];'
INIT = ' for(integer k=0;k<=58;k=k+1)begin de_delay[k]=0;rgb_delay[k]=0;frame_delay[k]=0;x_delay[k]=0;y_delay[k]=0;end'
SHIFT = ''' for(integer k=58;k>0;k=k-1)begin rgb_delay[k]=rgb_delay[k-1];de_delay[k]=de_delay[k-1];frame_delay[k]=frame_delay[k-1];x_delay[k]=x_delay[k-1];y_delay[k]=y_delay[k-1];end
 rgb_delay[0]={dut.hdmi_tx_rdata,dut.hdmi_tx_gdata,dut.hdmi_tx_bdata};de_delay[0]=dut.hdmi_tx_de && dut.hdmi_source_frame_active;
 frame_delay[0]=dut.hdmi_source_frame_id;x_delay[0]=dut.hdmi_timing_x;y_delay[0]=dut.hdmi_timing_y;'''


def transform(text):
    for expected in (DECL, INIT, SHIFT):
        assert text.count(expected)==1, 'TMDS checker template drift'
    text=text.replace(DECL, 'reg [72:0] monitor_fifo[0:58];integer monitor_index=0;reg [72:0] monitor_delayed=0;')
    text=text.replace(INIT, ' for(integer k=0;k<=58;k=k+1)monitor_fifo[k]=0;')
    text=text.replace(SHIFT, ''' monitor_fifo[monitor_index]={{dut.hdmi_tx_rdata,dut.hdmi_tx_gdata,dut.hdmi_tx_bdata},dut.hdmi_tx_de && dut.hdmi_source_frame_active,dut.hdmi_source_frame_id,dut.hdmi_timing_x,dut.hdmi_timing_y};
 monitor_index=monitor_index==58 ? 0:monitor_index+1;
 monitor_delayed=monitor_fifo[monitor_index];''')
    for name,field in [('rgb','72:49'),('de','48'),('frame','47:32'),('x','31:16'),('y','15:0')]:
        text=text.replace(f'{name}_delay[58]',f'monitor_delayed[{field}]')
        assert f'{name}_delay[' not in text
    return text
