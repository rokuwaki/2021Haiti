# GMT - The Generic Mapping Tools, Version 6.1.1

gmt gmtset MAP_FRAME_TYPE=plain
in_shadow="tmp.grd"
out_ps="topo.ps"
R1=-74.399071/-73.288628/18.159891/18.660241/-3/3
R2=-74.399071/-73.288628/18.159891/18.660241/-3000/3000
pers=105/25

# The file volume is very large: ~134MB
# gmt grdcut @earth_relief_01s_g -Gearth_at_01s.grd -R-75/-72/17/19
in_grd="earth_at_01s.grd"

gmt makecpt -T-2500/0/1 -Cgray -Z | awk 'NR <= 2500 {print $0}' > tmp.cpt
gmt makecpt -T0/2500/1 -Celevation -Z | awk '{print $0}' > tmp1
cat tmp1 >> tmp.cpt

gmt grdgradient $in_grd -G$in_shadow -A315 -N3
gmt psxy -R0/1/0/1 -JX1c -T -K -P > $out_ps
gmt psbasemap -R$R1 -JM5i -JZ1i -O -K -p$pers -Bx0.5 -By0.2 -Bz2+lkm -Bwsen >> $out_ps
gmt grdview $in_grd -R$R2 -I$in_shadow -J -JZ -O -K -p -Ctmp.cpt -N-3000+ggray90 -Qi500 >> $out_ps
echo "-73.475 18.408 0" > tmp.txt
gmt grdtrack -G$in_grd tmp.txt | gmt psxyz -R$R2 -J -JZ -p -Sc0.2 -Gwhite -O -K >> $out_ps
gmt psxy -R0/1/0/1 -JX1c -O -T >> $out_ps
gmt psconvert $out_ps -A -P -Tf -E600
