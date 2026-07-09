wus: This model is used for western North America and the Gulf Coastal Plain.
https://www.eas.slu.edu/eqc/eqc_mt/MECH.NA/MECHFIG/mech.html
https://doi.org/10.1785/0120110095

ttx: This model was developed for events near Timpson, Texas. It was required because of the deep column of Gulf Coast sediments.

Both of these models were taken from here:
https://www.eas.slu.edu/eqc/eqc_mt/MECH.NA/MECHFIG/mech.html
Find ttx and wus and click to see the models given by Herrmann.

Important Note on Attenuation (Q):
Keep in mind that Herrmann's Qp and Qs columns are actually inverse quality factors (1/Qp and 1/Qs). Therefore, we have to divide 1 by the value in Herrmann's attenuation columns to extract the actual Q values we should use in FK.

A value of 0 in Herrmann's table indicates very large Qp and Qs values (effectively infinite), meaning there is zero attenuation. Because the FK code cannot accept 0 for Q, I wrote 500/1000 for those specific layers to represent this lack of attenuation.

