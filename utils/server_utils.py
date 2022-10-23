#from turtle import distance
from pyparsing import countedArray
import geocoder
from nltk.tokenize import sent_tokenize
from nltk import download
import language_tool_python 
import utils.deeplearning_utils as ml
import utils.custom_utils as cutils

import glob
import PIL	
import os

def score_q2(province_text, country_text, continent_text, program_text, day_text, month_text, year_text, date_text, season_text, city_text):
	country_dict = {
	'AF': 'AFGHANISTAN',
	'AL': 'ALBANIA',
	'DZ': 'ALGERIA',
	'AS': 'AMERICAN SAMOA',
	'AD': 'ANDORRA',
	'AO': 'ANGOLA',
	'AI': 'ANGUILLA',
	'AQ': 'ANTARCTICA',
	'AG': 'ANTIGUA AND BARBUDA',
	'AR': 'ARGENTINA',
	'AM': 'ARMENIA',
	'AW': 'ARUBA',
	'AU': 'AUSTRALIA',
	'AT': 'AUSTRIA',
	'AZ': 'AZERBAIJAN',
	'BS': 'BAHAMAS',
	'BH': 'BAHRAIN',
	'BD': 'BANGLADESH',
	'BB': 'BARBADOS',
	'BY': 'BELARUS',
	'BE': 'BELGIUM',
	'BZ': 'BELIZE',
	'BJ': 'BENIN',
	'BM': 'BERMUDA',
	'BT': 'BHUTAN',
	'BO': 'BOLIVIA, PLURINATIONAL STATE OF',
	'BQ': 'BONAIRE, SINT EUSTATIUS AND SABA',
	'BA': 'BOSNIA AND HERZEGOVINA',
	'BW': 'BOTSWANA',
	'BV': 'BOUVET ISLAND',
	'BR': 'BRAZIL',
	'IO': 'BRITISH INDIAN OCEAN TERRITORY',
	'BN': 'BRUNEI DARUSSALAM',
	'BG': 'BULGARIA',
	'BF': 'BURKINA FASO',
	'BI': 'BURUNDI',
	'KH': 'CAMBODIA',
	'CM': 'CAMEROON',
	'CA': 'CANADA',
	'CV': 'CAPE VERDE',
	'KY': 'CAYMAN ISLANDS',
	'CF': 'CENTRAL AFRICAN REPUBLIC',
	'TD': 'CHAD',
	'CL': 'CHILE',
	'CN': 'CHINA',
	'CX': 'CHRISTMAS ISLAND',
	'CC': 'COCOS (KEELING) ISLANDS',
	'CO': 'COLOMBIA',
	'KM': 'COMOROS',
	'CG': 'CONGO',
	'CD': 'CONGO, THE DEMOCRATIC REPUBLIC OF THE',
	'CK': 'COOK ISLANDS',
	'CR': 'COSTA RICA',
	'HR': 'CROATIA',
	'CU': 'CUBA',
	'CY': 'CYPRUS',
	'CZ': 'CZECH REPUBLIC',
	'DK': 'DENMARK',
	'DJ': 'DJIBOUTI',
	'DM': 'DOMINICA',
	'DO': 'DOMINICAN REPUBLIC',
	'EC': 'ECUADOR',
	'EG': 'EGYPT',
	'SV': 'EL SALVADOR',
	'GQ': 'EQUATORIAL GUINEA',
	'ER': 'ERITREA',
	'EE': 'ESTONIA',
	'ET': 'ETHIOPIA',
	'FK': 'FALKLAND ISLANDS (MALVINAS)',
	'FO': 'FAROE ISLANDS',
	'FJ': 'FIJI',
	'FI': 'FINLAND',
	'FR': 'FRANCE',
	'GF': 'FRENCH GUIANA',
	'PF': 'FRENCH POLYNESIA',
	'TF': 'FRENCH SOUTHERN TERRITORIES',
	'GA': 'GABON',
	'GM': 'GAMBIA',
	'GE': 'GEORGIA',
	'DE': 'GERMANY',
	'GH': 'GHANA',
	'GI': 'GIBRALTAR',
	'GR': 'GREECE',
	'GL': 'GREENLAND',
	'GD': 'GRENADA',
	'GP': 'GUADELOUPE',
	'GU': 'GUAM',
	'GT': 'GUATEMALA',
	'GG': 'GUERNSEY',
	'GN': 'GUINEA',
	'GW': 'GUINEA-BISSAU',
	'GY': 'GUYANA',
	'HT': 'HAITI',
	'HM': 'HEARD ISLAND AND MCDONALD ISLANDS',
	'VA': 'HOLY SEE (VATICAN CITY STATE)',
	'HN': 'HONDURAS',
	'HK': 'HONG KONG',
	'HU': 'HUNGARY',
	'IS': 'ICELAND',
	'IN': 'INDIA',
	'ID': 'INDONESIA',
	'IR': 'IRAN, ISLAMIC REPUBLIC OF',
	'IQ': 'IRAQ',
	'IE': 'IRELAND',
	'IM': 'ISLE OF MAN',
	'IL': 'ISRAEL',
	'IT': 'ITALY',
	'JM': 'JAMAICA',
	'JP': 'JAPAN',
	'JE': 'JERSEY',
	'JO': 'JORDAN',
	'KZ': 'KAZAKHSTAN',
	'KE': 'KENYA',
	'KI': 'KIRIBATI',
	'KP': 'KOREA, DEMOCRATIC PEOPLE\'S REPUBLIC OF',
	'KR': 'KOREA, REPUBLIC OF',
	'KW': 'KUWAIT',
	'KG': 'KYRGYZSTAN',
	'LA': 'LAO PEOPLE\'S DEMOCRATIC REPUBLIC',
	'LV': 'LATVIA',
	'LB': 'LEBANON',
	'LS': 'LESOTHO',
	'LR': 'LIBERIA',
	'LY': 'LIBYAN ARAB JAMAHIRIYA',
	'LI': 'LIECHTENSTEIN',
	'LT': 'LITHUANIA',
	'LU': 'LUXEMBOURG',
	'MO': 'MACAO',
	'MK': 'MACEDONIA, THE FORMER YUGOSLAV REPUBLIC OF',
	'MG': 'MADAGASCAR',
	'MW': 'MALAWI',
	'MY': 'MALAYSIA',
	'MV': 'MALDIVES',
	'ML': 'MALI',
	'MT': 'MALTA',
	'MH': 'MARSHALL ISLANDS',
	'MQ': 'MARTINIQUE',
	'MR': 'MAURITANIA',
	'MU': 'MAURITIUS',
	'YT': 'MAYOTTE',
	'MX': 'MEXICO',
	'FM': 'MICRONESIA, FEDERATED STATES OF',
	'MD': 'MOLDOVA, REPUBLIC OF',
	'MC': 'MONACO',
	'MN': 'MONGOLIA',
	'ME': 'MONTENEGRO',
	'MS': 'MONTSERRAT',
	'MA': 'MOROCCO',
	'MZ': 'MOZAMBIQUE',
	'MM': 'MYANMAR',
	'NA': 'NAMIBIA',
	'NR': 'NAURU',
	'NP': 'NEPAL',
	'NL': 'NETHERLANDS',
	'NC': 'NEW CALEDONIA',
	'NZ': 'NEW ZEALAND',
	'NI': 'NICARAGUA',
	'NE': 'NIGER',
	'NG': 'NIGERIA',
	'NU': 'NIUE',
	'NF': 'NORFOLK ISLAND',
	'MP': 'NORTHERN MARIANA ISLANDS',
	'NO': 'NORWAY',
	'OM': 'OMAN',
	'PK': 'PAKISTAN',
	'PW': 'PALAU',
	'PS': 'PALESTINIAN TERRITORY, OCCUPIED',
	'PA': 'PANAMA',
	'PG': 'PAPUA NEW GUINEA',
	'PY': 'PARAGUAY',
	'PE': 'PERU',
	'PH': 'PHILIPPINES',
	'PN': 'PITCAIRN',
	'PL': 'POLAND',
	'PT': 'PORTUGAL',
	'PR': 'PUERTO RICO',
	'QA': 'QATAR',
	'RO': 'ROMANIA',
	'RU': 'RUSSIAN FEDERATION',
	'RW': 'RWANDA',
	'SH': 'SAINT HELENA, ASCENSION AND TRISTAN DA CUNHA',
	'KN': 'SAINT KITTS AND NEVIS',
	'LC': 'SAINT LUCIA',
	'MF': 'SAINT MARTIN (FRENCH PART)',
	'PM': 'SAINT PIERRE AND MIQUELON',
	'VC': 'SAINT VINCENT AND THE GRENADINES',
	'WS': 'SAMOA',
	'SM': 'SAN MARINO',
	'ST': 'SAO TOME AND PRINCIPE',
	'SA': 'SAUDI ARABIA',
	'SN': 'SENEGAL',
	'RS': 'SERBIA',
	'SC': 'SEYCHELLES',
	'SL': 'SIERRA LEONE',
	'SG': 'SINGAPORE',
	'SX': 'SINT MAARTEN (DUTCH PART)',
	'SK': 'SLOVAKIA',
	'SI': 'SLOVENIA',
	'SB': 'SOLOMON ISLANDS',
	'SO': 'SOMALIA',
	'ZA': 'SOUTH AFRICA',
	'GS': 'SOUTH GEORGIA AND THE SOUTH SANDWICH ISLANDS',
	'SS': 'SOUTH SUDAN',
	'ES': 'SPAIN',
	'LK': 'SRI LANKA',
	'SD': 'SUDAN',
	'SR': 'SURINAME',
	'SJ': 'SVALBARD AND JAN MAYEN',
	'SZ': 'SWAZILAND',
	'SE': 'SWEDEN',
	'CH': 'SWITZERLAND',
	'SY': 'SYRIAN ARAB REPUBLIC',
	'TW': 'TAIWAN, PROVINCE OF CHINA',
	'TJ': 'TAJIKISTAN',
	'TZ': 'TANZANIA, UNITED REPUBLIC OF',
	'TH': 'THAILAND',
	'TL': 'TIMOR-LESTE',
	'TG': 'TOGO',
	'TK': 'TOKELAU',
	'TO': 'TONGA',
	'TT': 'TRINIDAD AND TOBAGO',
	'TN': 'TUNISIA',
	'TR': 'TURKEY',
	'TM': 'TURKMENISTAN',
	'TC': 'TURKS AND CAICOS ISLANDS',
	'TV': 'TUVALU',
	'UG': 'UGANDA',
	'UA': 'UKRAINE',
	'AE': 'UNITED ARAB EMIRATES',
	'GB': 'UNITED KINGDOM',
	'US': 'UNITED STATES',
	'UM': 'UNITED STATES MINOR OUTLYING ISLANDS',
	'UY': 'URUGUAY',
	'UZ': 'UZBEKISTAN',
	'VU': 'VANUATU',
	'VE': 'VENEZUELA, BOLIVARIAN REPUBLIC OF',
	'VN': 'VIET NAM',
	'VG': 'VIRGIN ISLANDS, BRITISH',
	'VI': 'VIRGIN ISLANDS, U.S.',
	'WF': 'WALLIS AND FUTUNA',
	'EH': 'WESTERN SAHARA',
	'YE': 'YEMEN',
	'ZM': 'ZAMBIA',
	'ZW': 'ZIMBABWE',
}
	q2_score = 0
	def getLocation(userCity, userCountry, q2_score):
		try:
			locationIP = geocoder.ip('me')
			lat, lng = locationIP.latlng
			distanceKM = 0
			if city_text is not '':
				locationGiven = geocoder.arcgis(city_text)
				userCity = (locationGiven.latlng[0], locationGiven.latlng[1])			
				userIP = (locationIP.latlng[0], locationIP.latlng[1])
				distanceKM = distance(userIP, userCity)
				
			#--------------Continent Identificaiton---------------#
			if country_dict[locationIP.country].lower() == 'canada' or country_dict[locationIP.country].lower() == 'united states':
				true_continent = 'North America'
			else:
				true_continent = 'Other'
			
			province = locationIP.state.lower()
			country = country_dict[locationIP.country].lower()
			
			if distanceKM < 100 and distanceKM > 0:
				q2_score += 1
				print('distance = 1')
			else:
				print('distance = 0')
				pass
			if province.capitalize() == province_text: #should be nullproof
				q2_score += 1
				print('province = 1')
			else:
				print('province = 0')
				pass
			if country.upper() == country_text.upper(): #should be nullproof
				q2_score += 1
				print('country = 1')
			else:
				print('country = 0')
				pass
			if true_continent == continent_text: #should be nullproof
				q2_score += 1
				print('continent = 1')
			else:
				print('continent = 0')
				pass
			if 'Program' == program_text:
				q2_score += 1
				print('program = 1')
			else:
				print('program = 0')
		except:
			print('error: GeoCoder failed; suspect no internet access')
		return q2_score
				
	def distance(origin, destination):
		import math
		try:
			lat1, lon1 = origin
			lat2, lon2 = destination
			radius = 6371  # km
			dlat = math.radians(lat2-lat1)
			dlon = math.radians(lon2-lon1)
			a = math.sin(dlat/2) * math.sin(dlat/2) + math.cos(math.radians(lat1)) \
					* math.cos(math.radians(lat2)) * math.sin(dlon/2) * math.sin(dlon/2)
			c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
			d = radius * c
			return d
		except:
			print('error: distance function: failed to calculate distance, internet and geolocator likely offline')
	
	def date_and_time(q2_score):
		from datetime import date
		try:
			y_m_d = date.today()
			weekday = y_m_d.isoweekday()
			if weekday == 1:
				weekday = 'Monday'
			elif weekday == 2:
				weekday = 'Tuesday'
			elif weekday == 3:
				weekday = 'Wednesday'
			elif weekday == 4:
				weekday = 'Thurdsay'
			elif weekday == 5:
				weekday = 'Friday'
			elif weekday == 6:
				weekday = 'Saturday'
			elif weekday == 7:
				weekday='Sunday'
		except:
			print('error: weekday calculate failed; datetime nonfunctional')
		pass
		
		try:
			#Timezone identifier for Northern hemisphere
			doy = date.today().timetuple().tm_yday
			spring = range(80, 172)
			summer = range(172, 264)
			fall = range(264, 355)
		except:
			print('error: timezone identification failed, dateime nonfunctional')
		
		#This is not try except proofed, but it works error-free nonetheless.
		if doy in spring:
			season = 'Spring'
		elif doy in summer:
			season = 'Summer'
		elif doy in fall:
			season = 'Autumn'
		else:
			season = 'Winter'		

		if weekday == day_text: #is nullproof
			q2_score += 1	
			print('weekday = 1')
		else:
			print('weekdey = 0')	
		if y_m_d.strftime("%B") == month_text: #is nullproof
			q2_score += 1
			print('month = 1')
		else:
			print('month = 0')
		if str(y_m_d.year) == year_text: #is nullproof
			q2_score += 1
			print('year = 1')
		else:
			print('year = 0')
		if str(y_m_d.day) == date_text: #is nullproof
			q2_score += 1
			print('date = 1')
		else:
			print('date = 0')
		if  season == season_text: #is nullproof
			q2_score += 1
			print('season = 1')
		else:
			print('season = 0')
		return q2_score
		pass

	q2_component_one = getLocation(city_text, country_text, q2_score)
	q2_component_two = date_and_time(q2_score)
	q2_score = q2_component_one + q2_component_two
	print('q2_score: ', q2_score)
	return q2_score

# def q3_score(audio_bytes):
# 	print('q3 text: ', text)
# 	if text.count('lemon') != 0 or text.count('lime') != 0:
# 		if self.lemon:
# 			self.q3_score = self.q3_score + 1
# 			self.lemon = False
# 	if text.count('key') != 0 or text.count('chi') != 0 or text.count('Chi') != 0 or text.count('tea') != 0 and self.key:
# 		if self.key:
# 			self.q3_score = self.q3_score + 1
# 			self.key = False
# 	if text.count('ball') != 0 or text.count('bawl') != 0 and self.ball:
# 		if self.ball:
# 			self.q3_score = self.q3_score + 1
# 			self.ball = False
# 	print('q3 score: ' + str(self.q3_score))

def score_q4(one, two, three, four, five):
	q4_answers = []
	if one != 'string':
		q4_answers.append(100 - int(one))
	else:
		pass
	if two != 'string':
		print(two)
		q4_answers.append(int(one) - int(two))
	else:
		pass
	if three != 'string':
		q4_answers.append(int(two) - int(three))
	else:
		pass
	if four != 'string':
		q4_answers.append(int(three) - int(four))		
	else:
		pass
	if five != 'string':
		q4_answers.append(int(four) - int(five))			
	else:
		pass
	#Max 5 points
	q4_score = q4_answers.count(7)
	print('q4 score: ' + str(q4_score))
	return q4_score

def score_q10(sentence):
	q10_score = 0
	text = sentence	
	CorrectGrammar = True
	if text != 'string':
		try:
			sentenceList = sent_tokenize(text)
		except LookupError:
			print("Network error 'punkt' not downloaded")
		try:
			tool = language_tool_python.LanguageTool('en-US')
		except language_tool_python.server.requests.exceptions.ConnectionError:
			print("Network error 'language tool' not downloaded")
		if len(sentenceList) == 2:
			q10_score = q10_score + 1
		for sentence in sentenceList: #'recurs' through the identified sentences, checks their grammar. If any errors, sets  correctgrammar to False.
			if len(tool.check(sentence)) != 0:
				CorrectGrammar = False
		if CorrectGrammar == True:
			q10_score = q10_score + 1
		pass
	print('q10 score: ' + str(q10_score))
	return q10_score

def score_q16():
	q16_score = 0
	cube_score = 0
	infinity_score = 0
	clock_score = 0
	#looks good
	images = glob.glob("*.png")
	for infile in images:
		f, e = os.path.splitext(infile)
		outfile = f + ".jpg"
		if infile != outfile:
			im = PIL.Image.open(infile)
			im.size
			im.getbbox()
			im = im.crop(im.getbbox())
			bg = PIL.Image.new("RGB", im.size, (255, 255, 255))
			bg.paste(im, im)
			bg.save(outfile)
	#looks good

	#should work
	execution_path = os.getcwd()
	prediction = ml.CustomImagePrediction()
	prediction.setModelTypeAsSqueezeNet()
	prediction.setModelPath(os.path.join(execution_path, "model_ex-107_acc-0.990489.h5")) #path might be wrong
	prediction.setJsonPath(os.path.join(execution_path, "model_class.json")) #path might be wrong
	prediction.loadModel(prediction_speed='fastest', num_objects=4)
	#should work

	#should work
	imagenames = ['infinity.jpg', 'cube.jpg', 'clock.jpg']
	predictions = []
	probabilities = []
	for i in range(len(imagenames)):
		model_results = prediction.predictImage(os.path.join(execution_path, imagenames[int(i)]), result_count=1)
		print(model_results)
		probability_value = model_results[1]
		if int(probability_value[0]) < 0.95:
			predictions.append('other')
		else:
			predictions.append(model_results[0])
		probabilities.append(model_results[1])
		pass
	image_predictions = predictions
	image_probabilities = probabilities

	print(image_predictions)
	print(image_probabilities)
	infinity_prediction = image_predictions[0]
	infinity_probability = image_probabilities[0]	
	if infinity_prediction[0] == 'infinity' and int(infinity_probability[0]) >= 50:
		infinity_score = 1
	print('q16 score: ' + str(infinity_score))
		
	cube_prediction = image_predictions[1]
	cube_probability = image_probabilities[1]	
	if cube_prediction[0] == 'cube' and int(cube_probability[0]) >= 50:
		cube_score = 2
	print('q16b score: ' + str(cube_score))
		
	clock_prediction = image_predictions[2]
	clock_probability = image_probabilities[2]
	if clock_prediction[0] == 'clock' and int(clock_probability[0]) >= 50:
		clock_score = 5
	q16_score = infinity_score + cube_score + clock_score
	print('q16c score: ' + str(clock_score))
	return q16_score
	#LGTM

def score_q20(one, two, three, four, five, six, seven, eight):
	q20_score = 0
	if one == "Harry" and two  == "Barnes":
		q20_score += 1
	if three == "73":
		q20_score += 1		
	if four == "Orchard" and five == "Close":
		q20_score += 1
	if six == "Kingsbridge":
		q20_score += 1
	if seven == "Devon":
		q20_score += 1	
	if eight == '7':
		q20_score = 5
	print('q20 score: ' + str(q20_score))
	return q20_score