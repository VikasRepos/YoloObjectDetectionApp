import streamlit as st

st.set_page_config(page_title="Home",layout='wide')
st.title("YOLO V5 Object Detection App")
st.caption('This web application demostrate Object Detection')

# Content
st.markdown("""
### This App detects 196 car models from Image,
-[Click here for App](/YOLO_for_image/)
            
Below are the car models the YOLO V5 model will detect and classify
1. AM General Hummer SUV 2000
2. Acura RL Sedan 2012
3. Acura TL Sedan 2012
4. Acura TL Type-S 2008
5. Acura TSX Sedan 2012
6. Acura Integra Type R 2001
7. Acura ZDX Hatchback 2012
8. Aston Martin V8 Vantage Convertible 2012
9. Aston Martin V8 Vantage Coupe 2012
10. Aston Martin Virage Convertible 2012
11. Aston Martin Virage Coupe 2012
12. Audi RS 4 Convertible 2008
13. Audi A5 Coupe 2012
14. Audi TTS Coupe 2012
15. Audi R8 Coupe 2012
16. Audi V8 Sedan 1994
17. Audi 100 Sedan 1994
18. Audi 100 Wagon 1994
19. Audi TT Hatchback 2011
20. Audi S6 Sedan 2011
21. Audi S5 Convertible 2012
22. Audi S5 Coupe 2012
23. Audi S4 Sedan 2012
24. Audi S4 Sedan 2007
25. Audi TT RS Coupe 2012
26. BMW ActiveHybrid 5 Sedan 2012
27. BMW 1 Series Convertible 2012
28. BMW 1 Series Coupe 2012
29. BMW 3 Series Sedan 2012
30. BMW 3 Series Wagon 2012
31. BMW 6 Series Convertible 2007
32. BMW X5 SUV 2007
33. BMW X6 SUV 2012
34. BMW M3 Coupe 2012
35. BMW M5 Sedan 2010
36. BMW M6 Convertible 2010
37. BMW X3 SUV 2012
38. BMW Z4 Convertible 2012
39. Bentley Continental Supersports Conv. Convertible 2012
40. Bentley Arnage Sedan 2009
41. Bentley Mulsanne Sedan 2011
42. Bentley Continental GT Coupe 2012
43. Bentley Continental GT Coupe 2007
44. Bentley Continental Flying Spur Sedan 2007
45. Bugatti Veyron 16.4 Convertible 2009
46. Bugatti Veyron 16.4 Coupe 2009
47. Buick Regal GS 2012
48. Buick Rainier SUV 2007
49. Buick Verano Sedan 2012
50. Buick Enclave SUV 2012
51. Cadillac CTS-V Sedan 2012
52. Cadillac SRX SUV 2012
53. Cadillac Escalade EXT Crew Cab 2007
54. Chevrolet Silverado 1500 Hybrid Crew Cab 2012
55. Chevrolet Corvette Convertible 2012
56. Chevrolet Corvette ZR1 2012
57. Chevrolet Corvette Ron Fellows Edition Z06 2007
58. Chevrolet Traverse SUV 2012
59. Chevrolet Camaro Convertible 2012
60. Chevrolet HHR SS 2010
61. Chevrolet Impala Sedan 2007
62. Chevrolet Tahoe Hybrid SUV 2012
63. Chevrolet Sonic Sedan 2012
64. Chevrolet Express Cargo Van 2007
65. Chevrolet Avalanche Crew Cab 2012
66. Chevrolet Cobalt SS 2010
67. Chevrolet Malibu Hybrid Sedan 2010
68. Chevrolet TrailBlazer SS 2009
69. Chevrolet Silverado 2500HD Regular Cab 2012
70. Chevrolet Silverado 1500 Classic Extended Cab 2007
71. Chevrolet Express Van 2007
72. Chevrolet Monte Carlo Coupe 2007
73. Chevrolet Malibu Sedan 2007
74. Chevrolet Silverado 1500 Extended Cab 2012
75. Chevrolet Silverado 1500 Regular Cab 2012
76. Chrysler Aspen SUV 2009
77. Chrysler Sebring Convertible 2010
78. Chrysler Town and Country Minivan 2012
79. Chrysler 300 SRT-8 2010
80. Chrysler Crossfire Convertible 2008
81. Chrysler PT Cruiser Convertible 2008
82. Daewoo Nubira Wagon 2002
83. Dodge Caliber Wagon 2012
84. Dodge Caliber Wagon 2007
85. Dodge Caravan Minivan 1997
86. Dodge Ram Pickup 3500 Crew Cab 2010
87. Dodge Ram Pickup 3500 Quad Cab 2009
88. Dodge Sprinter Cargo Van 2009
89. Dodge Journey SUV 2012
90. Dodge Dakota Crew Cab 2010
91. Dodge Dakota Club Cab 2007
92. Dodge Magnum Wagon 2008
93. Dodge Challenger SRT8 2011
94. Dodge Durango SUV 2012
95. Dodge Durango SUV 2007
96. Dodge Charger Sedan 2012
97. Dodge Charger SRT-8 2009
98. Eagle Talon Hatchback 1998
99. FIAT 500 Abarth 2012
100. FIAT 500 Convertible 2012
101. Ferrari FF Coupe 2012
102. Ferrari California Convertible 2012
103. Ferrari 458 Italia Convertible 2012
104. Ferrari 458 Italia Coupe 2012
105. Fisker Karma Sedan 2012
106. Ford F-450 Super Duty Crew Cab 2012
107. Ford Mustang Convertible 2007
108. Ford Freestar Minivan 2007
109. Ford Expedition EL SUV 2009
110. Ford Edge SUV 2012
111. Ford Ranger SuperCab 2011
112. Ford GT Coupe 2006
113. Ford F-150 Regular Cab 2012
114. Ford F-150 Regular Cab 2007
115. Ford Focus Sedan 2007
116. Ford E-Series Wagon Van 2012
117. Ford Fiesta Sedan 2012
118. GMC Terrain SUV 2012
119. GMC Savana Van 2012
120. GMC Yukon Hybrid SUV 2012
121. GMC Acadia SUV 2012
122. GMC Canyon Extended Cab 2012
123. Geo Metro Convertible 1993
124. HUMMER H3T Crew Cab 2010
125. HUMMER H2 SUT Crew Cab 2009
126. Honda Odyssey Minivan 2012
127. Honda Odyssey Minivan 2007
128. Honda Accord Coupe 2012
129. Honda Accord Sedan 2012
130. Hyundai Veloster Hatchback 2012
131. Hyundai Santa Fe SUV 2012
132. Hyundai Tucson SUV 2012
133. Hyundai Veracruz SUV 2012
134. Hyundai Sonata Hybrid Sedan 2012
135. Hyundai Elantra Sedan 2007
136. Hyundai Accent Sedan 2012
137. Hyundai Genesis Sedan 2012
138. Hyundai Sonata Sedan 2012
139. Hyundai Elantra Touring Hatchback 2012
140. Hyundai Azera Sedan 2012
141. Infiniti G Coupe IPL 2012
142. Infiniti QX56 SUV 2011
143. Isuzu Ascender SUV 2008
144. Jaguar XK XKR 2012
145. Jeep Patriot SUV 2012
146. Jeep Wrangler SUV 2012
147. Jeep Liberty SUV 2012
148. Jeep Grand Cherokee SUV 2012
149. Jeep Compass SUV 2012
150. Lamborghini Reventon Coupe 2008
151. Lamborghini Aventador Coupe 2012
152. Lamborghini Gallardo LP 570-4 Superleggera 2012
153. Lamborghini Diablo Coupe 2001
154. Land Rover Range Rover SUV 2012
155. Land Rover LR2 SUV 2012
156. Lincoln Town Car Sedan 2011
157. MINI Cooper Roadster Convertible 2012
158. Maybach Landaulet Convertible 2012
159. Mazda Tribute SUV 2011
160. McLaren MP4-12C Coupe 2012
161. Mercedes-Benz 300-Class Convertible 1993
162. Mercedes-Benz C-Class Sedan 2012
163. Mercedes-Benz SL-Class Coupe 2009
164. Mercedes-Benz E-Class Sedan 2012
165. Mercedes-Benz S-Class Sedan 2012
166. Mercedes-Benz Sprinter Van 2012
167. Mitsubishi Lancer Sedan 2012
168. Nissan Leaf Hatchback 2012
169. Nissan NV Passenger Van 2012
170. Nissan Juke Hatchback 2012
171. Nissan 240SX Coupe 1998
172. Plymouth Neon Coupe 1999
173. Porsche Panamera Sedan 2012
174. Ram C/V Cargo Van Minivan 2012
175. Rolls-Royce Phantom Drophead Coupe Convertible 2012
176. Rolls-Royce Ghost Sedan 2012
177. Rolls-Royce Phantom Sedan 2012
178. Scion xD Hatchback 2012
179. Spyker C8 Convertible 2009
180. Spyker C8 Coupe 2009
181. Suzuki Aerio Sedan 2007
182. Suzuki Kizashi Sedan 2012
183. Suzuki SX4 Hatchback 2012
184. Suzuki SX4 Sedan 2012
185. Tesla Model S Sedan 2012
186. Toyota Sequoia SUV 2012
187. Toyota Camry Sedan 2012
188. Toyota Corolla Sedan 2012
189. Toyota 4Runner SUV 2012
190. Volkswagen Golf Hatchback 2012
191. Volkswagen Golf Hatchback 1991
192. Volkswagen Beetle Hatchback 2012
193. Volvo C30 Hatchback 2012
194. Volvo 240 Sedan 1993
195. Volvo XC90 SUV 2007
196. smart fortwo Convertible 2012
""")
            
            