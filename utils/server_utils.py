#from turtle import distance
from pyparsing import countedArray
import geocoder
from nltk.tokenize import sent_tokenize
from nltk import download
import language_tool_python 
import utils.deeplearning_utils as ml
import utils.custom_utils as cutils

import wave
import speech_recognition as sr
import pyaudio
import io

import glob
import PIL	
import os

def transcriber(audio_file):
	arb = sr.AudioFile(audio_file)
	r = sr.Recognizer()
	with arb as source:
		audiodata = r.record(arb)
		print('in arb')
	try:
		text = r.recognize_google(audiodata, language='en-US', show_all=False)
		print(text)
	except:
		print('Error: speech recognition; suspect network error.')
	return text

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

def score_q3(audio_file):
	q3_score = 0
	lemon = True ; key = True ; ball = True
	text = transcriber(audio_file)
	if text:
		print('recognized text:', text)
		if text.count('lemon') != 0 or text.count('lime') != 0:
			if lemon:
				q3_score = q3_score + 1
				lemon = False
		if text.count('key') != 0 or text.count('chi') != 0 or text.count('Chi') != 0 or text.count('tea') != 0 and key:
			if key:
				q3_score = q3_score + 1
				key = False
		if text.count('ball') != 0 or text.count('bawl') != 0 and ball:
			if ball:
				q3_score = q3_score + 1
				ball = False
		if text.count('kebal') != 0:
			q3_score = q3_score + 2
		if q3_score > 3:
			q3_score = 3
	print('q3 score: ' + str(q3_score))
	return q3_score 

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

def score_q5(audio_file):
	q5_score = 0
	lemon = True ; key = True ; ball = True
	text = transcriber(audio_file)
	print('q5 text: ', text)
	if text.count('lemon') != 0 or text.count('lime') != 0:
		if lemon:
			q5_score = q5_score + 1
			lemon = False
	if text.count('key') != 0 or text.count('chi') != 0 or text.count('Chi') != 0 or text.count('tea') != 0 and key:
		if key:
			q5_score = q5_score + 1
			key = False
	if text.count('ball') != 0 or text.count('bawl') != 0 and ball:
		if ball:
			q5_score = q5_score + 1
			ball = False
	print('q5 score: ' + str(q5_score))
	return q5_score

def score_q6(audio_file):
	p_words = True
	import regex
	from nltk.corpus import words
	from nltk import download
	try:
		download('words')		
	except:
		print('error: failed nltk words download')
	text = transcriber(audio_file)
	p_finder = regex.compile(r'p\w+')
	p_word_list = list(set(p_finder.findall(text)))
	print('here are all found P words:')
	print(p_word_list)
	
	for i in range(len(p_word_list)):
		if p_word_list[i] in words.words():
			word_count += 1
			pass
	word_count=int(word_count)
	if word_count <= 1:
		q6_score = 0
	elif word_count >1 and word_count <4:
		q6_score = 1
	elif word_count >3 and word_count <6:
		q6_score = 2
	elif word_count >5 and word_count <8:
		q6_score = 3
	elif word_count >7 and word_count <11:
		q6_score = 4
	elif word_count >10 and word_count <14:
		q6_score = 5		
	elif word_count >13 and word_count <18:
		q6_score = 6
	elif word_count >= 18:
		q6_score = 7	
	else:
		print('error: unknown error')
		#Max 7 points
	print('q6 score: ' + str(q6_score))
	return q6_score

def q6B_score(audio_file):
	import regex
	from nltk.corpus import words
	from nltk import download
	try:
		download('words')		
	except:
		print('error: failed nltk words download')
	animal_list = ['canidae', 'felidae', 'cat', 'cattle', 'dog', 'donkey', 'goat', 'guinea', 'pig', 'horse', 'pig', 'rabbit', 'fancy', 'rat', 'varieties', 'laboratory', 'rat', 'strains', 'sheep', 'breeds', 'water', 'buffalo', 'breeds', 'chicken', 'breeds', 'duck', 'breeds', 'goose', 'breeds', 'pigeon', 'breeds', 'turkey', 'breeds', 'aardvark', 'aardwolf', 'african', 'buffalo', 'african', 'elephant', 'african', 'leopard', 'albatross', 'alligator', 'alpaca', 'american', 'buffalo', 'bison', 'american', 'robin', 'amphibian', 'list', 'anaconda', 'angelfish', 'anglerfish', 'ant', 'anteater', 'antelope', 'antlion', 'ape', 'aphid', 'arabian', 'leopard', 'arctic', 'fox', 'arctic', 'wolf', 'armadillo', 'arrow', 'crab', 'asp', 'ass', 'donkey', 'baboon', 'badger', 'bald', 'eagle', 'bandicoot', 'barnacle', 'barracuda', 'basilisk', 'bass', 'bat', 'beaked', 'whale', 'bear', 'list', 'beaver', 'bedbug', 'bee', 'beetle', 'bird', 'list', 'bison', 'blackbird', 'black', 'panther', 'black', 'widow', 'spider', 'blue', 'bird', 'bluejay', 'blue', 'whale', 'boa', 'boar', 'bobcat', 'bobolink', 'bonobo', 'booby', 'box', 'jellyfish', 'bovid', 'buffalo', 'african', 'buffalo', 'american', 'bison', 'bug', 'butterfly', 'buzzard', 'camel', 'canid', 'cape', 'buffalo', 'capybara', 'cardinal', 'caribou', 'carp', 'cat', 'list', 'catshark', 'caterpillar', 'catfish', 'cattle', 'list', 'centipede', 'cephalopod', 'chameleon', 'cheetah', 'chickadee', 'chicken', 'chimpanzee', 'chinchilla', 'chipmunk', 'clam', 'clownfish', 'cobra', 'cockroach', 'cod', 'condor', 'constrictor', 'coral', 'cougar', 'cow', 'coyote', 'crab', 'crane', 'crane', 'fly', 'crawdad', 'crayfish', 'cricket', 'crocodile', 'crow', 'cuckoo', 'cicada', 'damselfly', 'deer', 'dingo', 'dinosaur', 'list', 'dog', 'list', 'dolphin', 'donkey', 'list', 'dormouse', 'dove', 'dragonfly', 'dragon', 'duck', 'dung', 'beetle', 'eagle', 'earthworm', 'earwig', 'echidna', 'eel', 'egret', 'elephant', 'elephant', 'seal', 'elk', 'emu', 'english', 'pointer', 'ermine', 'falcon', 'ferret', 'finch', 'firefly', 'fish', 'flamingo', 'flea', 'fly', 'flyingfish', 'fowl', 'fox', 'frog', 'fruit', 'bat', 'gamefowl', 'list', 'galliform', 'list', 'gazelle', 'gecko', 'gerbil', 'giant', 'panda', 'giant', 'squid', 'gibbon', 'gila', 'monster', 'giraffe', 'goat', 'list', 'goldfish', 'goose', 'list', 'gopher', 'gorilla', 'grasshopper', 'great', 'blue', 'heron', 'great', 'white', 'shark', 'grizzly', 'bear', 'ground', 'shark', 'ground', 'sloth', 'grouse', 'guan', 'list', 'guanaco', 'guineafowl', 'list', 'guinea', 'pig', 'list', 'gull', 'guppy', 'haddock', 'halibut', 'hammerhead', 'shark', 'hamster', 'hare', 'harrier', 'hawk', 'hedgehog', 'hermit', 'crab', 'heron', 'herring', 'hippopotamus', 'hookworm', 'hornet', 'horse', 'list', 'hoverfly', 'hummingbird', 'humpback', 'whale', 'hyena', 'iguana', 'impala', 'irukandji', 'jellyfish', 'jackal', 'jaguar', 'jay', 'jellyfish', 'junglefowl', 'kangaroo', 'kangaroo', 'mouse', 'kangaroo', 'rat', 'kingfisher', 'kite', 'kiwi', 'koala', 'koi', 'komodo', 'dragon', 'krill', 'ladybug', 'lamprey', 'landfowl', 'land', 'snail', 'lark', 'leech', 'lemming', 'lemur', 'leopard', 'leopon', 'limpet', 'lion', 'lizard', 'llama', 'lobster', 'locust', 'loon', 'louse', 'lungfish', 'lynx', 'macaw', 'mackerel', 'magpie', 'mammal', 'manatee', 'mandrill', 'manta', 'ray', 'marlin', 'marmoset', 'marmot', 'marsupial', 'marten', 'mastodon', 'meadowlark', 'meerkat', 'mink', 'minnow', 'mite', 'mockingbird', 'mole', 'mollusk', 'mongoose', 'monitor', 'lizard', 'monkey', 'moose', 'mosquito', 'moth', 'mountain', 'goat', 'mouse', 'mule', 'muskox', 'narwhal', 'newt', 'new', 'world', 'quail', 'nightingale', 'ocelot', 'octopus', 'old', 'world', 'quail', 'opossum', 'orangutan', 'orca', 'ostrich', 'otter', 'owl', 'ox', 'panda', 'panther', 'panthera', 'hybrid', 'parakeet', 'parrot', 'parrotfish', 'partridge', 'peacock', 'peafowl', 'pelican', 'penguin', 'perch', 'peregrine', 'falcon', 'pheasant', 'pig', 'pigeon', 'list', 'pike', 'pilot', 'whale', 'pinniped', 'piranha', 'planarian', 'platypus', 'polar', 'bear', 'pony', 'porcupine', 'porpoise', 'portuguese', 'man', 'o', 'war', 'possum', 'prairie', 'dog', 'prawn', 'praying', 'mantis', 'primate', 'ptarmigan', 'puffin', 'puma', 'python', 'quail', 'quelea', 'quokka', 'rabbit', 'list', 'raccoon', 'rainbow', 'trout', 'rat', 'rattlesnake', 'raven', 'ray', 'batoidea', 'ray', 'rajiformes', 'red', 'panda', 'reindeer', 'reptile', 'rhinoceros', 'right', 'whale', 'roadrunner', 'rodent', 'rook', 'rooster', 'roundworm', 'saber', 'toothed', 'cat', 'sailfish', 'salamander', 'salmon', 'sawfish', 'scale', 'insect', 'scallop', 'scorpion', 'seahorse', 'sea', 'lion', 'sea', 'slug', 'sea', 'snail', 'shark', 'list', 'sheep', 'list', 'shrew', 'shrimp', 'silkworm', 'silverfish', 'skink', 'skunk', 'sloth', 'slug', 'smelt', 'snail', 'snake', 'list', 'snipe', 'snow', 'leopard', 'sockeye', 'salmon', 'sole', 'sparrow', 'sperm', 'whale', 'spider', 'spider', 'monkey', 'spoonbill', 'squid', 'squirrel', 'starfish', 'star', 'nosed', 'mole', 'steelhead', 'trout', 'stingray', 'stoat', 'stork', 'sturgeon', 'sugar', 'glider', 'swallow', 'swan', 'swift', 'swordfish', 'swordtail', 'tahr', 'takin', 'tapir', 'tarantula', 'tarsier', 'tasmanian', 'devil', 'termite', 'tern', 'thrush', 'tick', 'tiger', 'tiger', 'shark', 'tiglon', 'toad', 'tortoise', 'toucan', 'trapdoor', 'spider', 'tree', 'frog', 'trout', 'tuna', 'turkey', 'list', 'turtle', 'tyrannosaurus', 'urial', 'vampire', 'bat', 'vampire', 'squid', 'vicuna', 'viper', 'vole', 'vulture', 'wallaby', 'walrus', 'wasp', 'warbler', 'water', 'boa', 'water', 'buffalo', 'weasel', 'whale', 'whippet', 'whitefish', 'whooping', 'crane', 'wildcat', 'wildebeest', 'wildfowl', 'wolf', 'wolverine', 'wombat', 'woodpecker', 'worm', 'wren', 'xerinae', 'x', 'ray', 'fish', 'yak', 'yellow', 'perch', 'zebra', 'zebra', 'finch', 'animals', 'by', 'number', 'of', 'neurons', 'animals', 'by', 'size', 'common', 'household', 'pests', 'common', 'names', 'of', 'poisonous', 'animals', 'alpaca', 'bali', 'cattle', 'cat', 'cattle', 'chicken', 'dog', 'domestic', 'bactrian', 'camel', 'domestic', 'canary', 'domestic', 'dromedary', 'camel', 'domestic', 'duck', 'domestic', 'goat', 'domestic', 'goose', 'domestic', 'guineafowl', 'domestic', 'hedgehog', 'domestic', 'pig', 'domestic', 'pigeon', 'domestic', 'rabbit', 'domestic', 'silkmoth', 'domestic', 'silver', 'fox', 'domestic', 'turkey', 'donkey', 'fancy', 'mouse', 'fancy', 'rat', 'lab', 'rat', 'ferret', 'gayal', 'goldfish', 'guinea', 'pig', 'guppy', 'horse', 'koi', 'llama', 'ringneck', 'dove', 'sheep', 'siamese', 'fighting', 'fish', 'society', 'finch', 'yak', 'water', 'buffalo']
	animal_word_finder = regex.compile(r'\w+')
	text = transcriber(audio_file)
	word_list = animal_word_finder.findall(text)
	print('identified words: ', set(word_list) & set(animal_list))
	word_count = len(set(word_list) & set(animal_list))
	print('number of correct words identified: ' + str(word_count))
	
	word_count = int(word_count)
	if word_count >0 and word_count <5:
		q6B_score = 0
	elif word_count >4 and word_count <7:
		q6B_score = 1
	elif word_count >6 and word_count <9:
		q6B_score = 2
	elif word_count >8 and word_count <11:
		q6B_score = 3
	elif word_count >10 and word_count <14:
		q6B_score = 4
	elif word_count >13 and word_count <17:
		q6B_score = 5		
	elif word_count >16 and word_count <22:
		q6B_score = 6
	elif word_count >= 22:
		q6B_score = 7
	else:
		print('error: unlabeled error occured')
	#Max 7 points
	print('q6B score: ' + str(q6B_score))	
	return q6B_score

def score_q7C(audio_file):
	q7C_score = 0
	harry = True; barnes = True; seventythree = True; orchard = True; close = True
	kingsbridge = True; devon = True
	text = transcriber(audio_file)
	if text.count('harry') != 0 or text.count('hairy') != 0 or text.count('hary') != 0 or text.count('hurry') != 0:
		if harry:
			harry = False
			q7C_score = q7C_score + 1
	if text.count('barnes') != 0 or text.count('barns') != 0 or text.count('borns') != 0 or text.count('bornes') != 0 or text.count('burns') != 0:
		if barnes:
			barnes = False
			q7C_score = q7C_score + 1
	if text.count('73') != 0 or text.count('seventy three') != 0 or text.count('seventy-three') != 0:
		if seventythree:
			seventythree = False
			q7C_score = q7C_score + 1	
	if text.count('orchard') != 0 or text.count('orcherd') != 0 or text.count('orchid') != 0:
		if orchard:
			orchard = False
			q7C_score = q7C_score + 1
	if text.count('close') != 0 or text.count('clothes') != 0 or text.count('cloves') != 0:
		if close:
			close = False
			q7C_score = q7C_score + 1
	if text.count('kingsbridge') != 0:
		if kingsbridge:
			kingsbridge = False
			q7C_score = q7C_score + 1	
	if text.count('devon') != 0 or text.count('devin') != 0 or text.count('devon') != 0:
		if devon:
			devon = False
			q7C_score = q7C_score + 1
	print('q7C score: ' + str(q7C_score))
	return q7C_score

def score_q8(audio_file):
	q8_score = 0
	firstname = True; lastname = True
	text = transcriber(audio_file)

	if text.count('justin') != 0 or text.count('dustin') != 0 or text.count('dustin') != 0 or text.count('justine') != 0:
		if firstname:				
			firstname = False
			q8_score = q8_score + 0.5
	if text.count('trudeau') != 0 or text.count('judeau') != 0 or text.count('judo') or text.count('to do') or text.count('trudell') or text.count('tudo') or text.count('truth'):
		if lastname:
			lastname = False
			q8_score = q8_score + 0.5
	if q8_score != 1:
		q8_score = 0
	print('q8 score: ' + str(q8_score))
	return q8_score

def score_q8B(audio_file):
	q8B_score = 0
	firstname = True; lastname = True
	text = transcriber(audio_file)
	if text.count('alexander') != 0 or text.count('alex') or text.count('john') or text.count('johnathan'):
		if firstname:				
			firstname = False
			q8B_score = q8B_score + 0.5
	if text.count('macdonald') != 0 or text.count('mcdonald') != 0 or text.count('mcconell')  or text.count('mcdonalds') or text.count("mcdonald's"):
		if lastname:
			lastname = False
			q8B_score = q8B_score + 0.5
	if q8B_score != 1:
		q8B_score = 0
	print('q8B score: ' + str(q8B_score))
	return q8B_score

def score_q8B(audio_file):
	q8B_score = 0
	firstname = True; lastname = True
	text = transcriber(audio_file)

	if text.count('barack') != 0 or text.count('brock') or text.count('barocque'):
		if firstname:				
			firstname = False
			q8C_score = q8C_score + 0.5
	if text.count('obama') != 0 or text.count('osama') != 0 or text.count('drama'):
		if lastname:
			lastname = False
			q8C_score = q8C_score + 0.5
	if q8C_score != 1:
		q8C_score = 0
	print('q8C score: ' + str(q8C_score))
	return q8C_score

def score_q8B(audio_file):
	q8B_score = 0
	firstname = True; lastname = True
	text = transcriber(audio_file)

	if text.count('joseph') != 0 or text.count('joe') or text.count('doe') or text.count('dough'):
		if firstname:				
			firstname = False
			q8D_score = q8D_score + 0.5
	if text.count('biden') != 0 or text.count('raiden'):
		if lastname:
			lastname = False
			q8D_score = q8D_score + 0.5
	if q8D_score != 1:
		q8D_score = 0
	else:
		q8D_score = 1
	print('q8D score: ' + str(q8D_score))
	return q8D_score

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

def score_q11(audio_file):
	q11_score = 0
	caterpillar = True
	text = transcriber(audio_file)

	if text.count('caterpillar') != 0:
		if caterpillar:
			caterpillar = False
			q11_score = q11_score + 1
	print('q11 score: ' + str(q11_score))
	return q11_score

def score_q11B(audio_file):
	q11B_score = 0
	eccentricity = True
	text = transcriber(audio_file)
	if text.count('eccentricity') != 0:
		if eccentricity:
			eccentricity = False
			q11B_score = q11B_score + 1
	print('q11B score: ' + str(q11B_score))
	return q11B_score

def score_q11C(audio_file):
	q11C_score = 0
	word = True
	text = transcriber(audio_file)
	if text.count('unintelligible') != 0:
		if word:
			word = False
			q11C_score = q11C_score + 1
	print('q11C score: ' + str(q11C_score))	
	return q11C_score

def score_q11D(audio_file):
	q11D_score = 0
	word = True
	text = transcriber(audio_file)
	if text.count('statistician') != 0:
		if word:
			word = False
			q11D_score = q11D_score + 1
	print('q11D score: ' + str(q11D_score))	
	return q11D_score

def score_q12(audio_file):
	q12_score = 0
	glitters = True
	text = transcriber(audio_file)
	if text.count('all that glitters is not gold') != 0 or text.count('all the glitters is not gold') != 0 or text.count('the glitters is not gold') != 0 or text.count('that glitters is not gold') != 0:
		if glitters:
			glitters = False
			q12_score = 1		
	print('q12 score: ' + str(q12_score))
	return q12_score

def score_q12B(audio_file):
	q12B_score = 0
	gold = True
	text = transcriber(audio_file)
	if text.count('a stitch in time saves nine') != 0 or text.count('stitch in time saves nine') != 0:
		if gold:
			gold = False
			q12B_score = 1		
	print('q12B score: ' + str(q12B_score))
	return q12B_score

def score_q13(audio_file):
	q13_score = 0
	spoon = True
	text = transcriber(audio_file)
	if text.count('spoon') != 0:
		if spoon:
			q13_score = q13_score + 1
			spoon = False
	print('q13 score: ' + str(q13_score))
	return q13_score

def score_q13B(audio_file):
	q13B_score = 0
	book = True
	text = transcriber(audio_file)
	if text.count('book') != 0 or text.count('novel') != 0:
		if book:
			q13B_score = q13B_score + 1
			book = False
	print('q13B score: ' + str(q13B_score))
	return q13B_score

def score_q13C(audio_file):
	q13C_score = 0
	kangaroo = True
	text = transcriber(audio_file)
	if text.count('kangaroo') != 0:
		if kangaroo:
			q13C_score = q13C_score + 1
			kangaroo = False
	print('q13C score: ' + str(q13C_score))
	return q13C_score

def score_q13D(audio_file):
	q13D_score = 0
	penguin = True
	text = transcriber(audio_file)
	if text.count('penguin') != 0:
		if penguin:
			q13D_score = q13D_score + 1
			penguin = False
	print('q13D score: ' + str(q13D_score))
	return q13D_score

def score_q13E(audio_file):
	q13E_score = 0
	anchor = True
	text = transcriber(audio_file)
	if text.count('anchor') != 0:
		if anchor:
			q13E_score = q13E_score + 1
			anchor = False
	print('q13E score: ' + str(q13E_score))
	return q13E_score

def score_q13F(audio_file):
	q13F_score = 0
	camel = True
	text = transcriber(audio_file)
	if text.count('camel') != 0 or text.count('dromedary') != 0:
		if camel:
			q13F_score = q13F_score + 1
			camel = False
	print('q13F score: ' + str(q13F_score))
	return q13F_score

def score_q13G(audio_file):
	q13G_score = 0
	harp = True
	text = transcriber(audio_file)
	if text.count('harp') != 0 or text.count('harpsicord') != 0:
		if harp:
			q13G_score = q13G_score + 1
			harp = False
	print('q13G score: ' + str(q13G_score))
	return q13G_score

def score_q13H(audio_file):
	q13H_score = 0
	rhino = True
	text = transcriber(audio_file)
	if text.count('rhino') != 0 or text.count('rhinoceros') != 0:
		if rhino:
			q13H_score = q13H_score + 1
			rhino = False
	print('q13H score: ' + str(q13H_score))
	return q13H_score

def score_q13I(audio_file):
	q13I_score = 0
	barrel = True
	text = transcriber(audio_file)
	if text.count('barrel') != 0 or text.count('cask') != 0:
		if barrel:
			q13I_score = q13I_score + 1
			barrel = False
	print('q13I score: ' + str(q13I_score))
	return q13I_score

def score_q13J(audio_file):
	q13J_score = 0
	crown = True
	text = transcriber(audio_file)
	if text.count('crown') != 0 or text.count('tiara') != 0:
		if crown:
			q13J_score = q13J_score + 1
			crown = False
	print('q13J score: ' + str(q13J_score))
	return q13J_score

def score_q13K(audio_file):
	q13K_score = 0
	alligator = True
	text = transcriber(audio_file)
	if text.count('alligator') != 0 or text.count('crocodile') != 0 or text.count('gator') != 0 or text.count('croc') != 0:
		if alligator:
			q13K_score = q13K_score + 1
			alligator = False
	print('q13K score: ' + str(q13K_score))
	return q13K_score

def score_q13L(audio_file):
	q13L_score = 0
	accordion = True
	text = transcriber(audio_file)
	if text.count('accordion') != 0 or text.count('squeezebox') != 0:
		if accordion:
			q13L_score = q13L_score + 1
			accordion = False
	print('q13L score: ' + str(q13L_score))
	return q13L_score

def score_q15(audio_file):
	q15_score = 0
	text = transcriber(audio_file)
	sew = True
	pint = True
	soot = True
	dough = True
	ht = True
	def on_text(self, instance, text):
		if text.count('so') != 0 or text.count('sew') != 0:
			if sew:
				q15_prescore = q15_prescore + 1
				sew = False		
		if text.count('pint') != 0:
			if pint:
				q15_prescore = q15_prescore + 1
				pint = False		
		if text.count('soot') != 0 or text.count('set') != 0 or text.count('suit') != 0 or text.count('sit') != 0 or text.count('suet') != 0 or text.count('search') != 0:
			if soot:
				q15_prescore = q15_prescore + 1
				soot = False
		if text.count('dough') != 0 or text.count('do') != 0 or text.count('go') != 0:
			if dough:
				q15_prescore = q15_prescore + 1
				dough = False
		if text.count('height') != 0 or text.count('tight') != 0:
			if ht:
				q15_prescore = q15_prescore + 1
				ht = False
		if q15_prescore == 5:
			q15_score = 1
		else:
			q15_score = 0
		print('q15 score: ', q15_score)
		return q15_score

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

def score_q17(audio_file, button_press):
	q17_score = 0
	text = transcriber(audio_file)
	eight = True
	if text.count('eight') != 0 or text.count('8') != 0:
		if eight and button_press == False:
			q17_score = q17_score + 1
			eight = False
	print('q17 score: ' + str(q17_score))
	return q17_score

def score_q17B(audio_file, button_press):
	q17B_score = 0
	text = transcriber(audio_file)
	ten = True
	if text.count('ten') != 0 or text.count('10') != 0:
		if ten and button_press == False:
			q17B_score = q17B_score + 1
			ten = False
	print('q17B score: ' + str(q17B_score))
	return q17B_score

def score_q17C(audio_file, button_press):
	q17C_score = 0
	text = transcriber(audio_file)
	seven = True
	if text.count('seven') != 0 or text.count('7') != 0:
		if seven and button_press == False:
			q17C_score = q17C_score + 1
			seven = False
	print('q17C score: ' + str(q17C_score))
	return q17C_score	

def score_q17D(audio_file, button_press):
	q17D_score = 0
	text = transcriber(audio_file)
	nine = True
	if text.count('nine') != 0 or text.count('9') != 0:
		if nine and button_press == False:
			q17D_score = q17D_score + 1
			nine = False
	print('q17D score: ' + str(q17D_score))
	return q17D_score

def score_q18(audio_file, button_press):
	q18_score = 0
	text = transcriber(audio_file)
	k = True
	if text.count('k') != 0 or text.count('kay') != 0 or text.count('okay') != 0:
		if k:
			q18_score = q18_score + 1
			k = False
	print('q18 score: ' + str(q18_score))
	return q18_score

def score_q18B(audio_file, button_press):
	q18B_score = 0
	text = transcriber(audio_file)
	m = True
	if text.count('m') != 0 or text.count('em') != 0:
		if m:
			q18B_score = q18B_score + 1
			m = False
	print('q18B score: ' + str(q18B_score))
	return q18B_score

def score_q18C(audio_file, button_press):
	q18C_score = 0
	text = transcriber(audio_file)
	a = True
	if text.count('a') != 0 or text.count('eh') != 0 or text.count('ay') != 0 or text.count('yay') != 0:
		if a:
			q18C_score = q18C_score + 1
			a = False
	print('q18C score: ' + str(q18C_score))
	return q18C_score

def score_q18D(audio_file, button_press):
	q18D_score = 0
	text = transcriber(audio_file)
	t = True
	if text.count('t') != 0 or text.count('tea') != 0 or text.count('tee') != 0 or text.count('chi') != 0:
		if t:
			q18D_score = q18D_score + 1
			t = False
	print('q18D score: ' + str(q18D_score))
	return q18D_score

def score_q19(audio_file, button_press):
	q19_score = 0
	text = transcriber(audio_file)
	harry = True
	barnes = True
	seventythree = True
	orchard = True
	close = True
	kingsbridge = True
	devon = True

	if text.count('harry') != 0 or text.count('hairy') != 0 or text.count('hary') != 0 or text.count('hurry') != 0:
		if harry:
			harry = False
			q19_score = q19_score + 1
	if text.count('barnes') != 0 or text.count('barns') != 0 or text.count('borns') != 0 or text.count('bornes') != 0 or text.count('burns') != 0:
		if barnes:
			barnes = False
			q19_score = q19_score + 1
	if text.count('73') != 0 or text.count('seventy three') != 0 or text.count('seventy-three') != 0:
		if seventythree:
			seventythree = False
			q19_score = q19_score + 1	
	if text.count('orchard') != 0 or text.count('orcherd') != 0 or text.count('orchid') != 0:
		if orchard:
			orchard = False
			q19_score = q19_score + 1
	if text.count('close') != 0 or text.count('clothes') != 0 or text.count('cloves') != 0:
		if close:
			close = False
			q19_score = q19_score + 1
	if text.count('kingsbridge') != 0:
		if kingsbridge:
			kingsbridge = False
			q19_score = q19_score + 1	
	if text.count('devon') != 0 or text.count('devin') != 0 or text.count('devon') != 0:
		if devon:
			devon = False
			q19_score = q19_score + 1
	print('q19 score: ' + str(q19_score))
	return q19_score

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