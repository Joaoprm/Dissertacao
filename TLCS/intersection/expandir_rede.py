import xml.etree.ElementTree as ET

input_net = "network_5_intersections_bus_lane.net.xml"
output_patch = "patch.edg.xml"

tree = ET.parse(input_net)
root = tree.getroot()

with open(output_patch, "w") as f:
    f.write('<edges>\n')
    for edge in root.findall('edge'):
        edge_id = edge.get('id')
        # Ignorar as conexões internas dos cruzamentos
        if edge_id and not edge_id.startswith(':'):
            lanes = edge.findall('lane')
            num_lanes_original = len(lanes)
            # Adicionamos apenas 1 via (a de BUS), mantendo o passeio original
            new_total = num_lanes_original + 1
            
            f.write(f'   <edge id="{edge_id}" numLanes="{new_total}">\n')
            
            # INDEX 0: O PASSEIO (Extrema Direita)
            f.write(f'      <lane index="0" allow="pedestrian" width="2.00"/>\n')
            
            # INDEX 1: A NOVA BUS LANE (Ao lado do passeio)
            f.write(f'      <lane index="1" allow="bus" width="3.20"/>\n')
            
            # INDEX 2 EM DIANTE: VIAS PARA CARROS
            for i in range(2, new_total):
                f.write(f'      <lane index="{i}" allow="all" width="3.20"/>\n')
                
            f.write(f'   </edge>\n')
    f.write('</edges>\n')

print("Patch gerado: Passeio na extremidade (index 0) e BUS lane no index 1.")