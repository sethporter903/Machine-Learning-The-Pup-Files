// Create a basic map object
var myMap = L.map("map", {
    center: [52.219456, -1.182953],
    zoom: 8
});

// Create a streets tile layer to the map
L.tileLayer("https://api.tiles.mapbox.com/v4/{id}/{z}/{x}/{y}.png?access_token={accessToken}", {
    attribution: "Map data &copy; <a href=\"https://www.openstreetmap.org/\">OpenStreetMap</a> contributors, <a href=\"https://creativecommons.org/licenses/by-sa/2.0/\">CC-BY-SA</a>, Imagery © <a href=\"https://www.mapbox.com/\">Mapbox</a>",
    maxZoom: 18,
    id: "mapbox.streets",
    accessToken: API_KEY
}).addTo(myMap);

var features = data.features;

var doggos = ['affenpinscher', 'afghan_hound', 'african_hunting_dog', 'airedale', 'alaskan_malamute', 
'american_staffordshire_terrier', 'appenzeller', 'australian_terrier', 'basenji', 'basset_hound', 'beagle', 
'bedlington_terrier', 'bernese_mountain_dog', 'black_and_tan_coonhound', 'blenheim_spaniel', 'bloodhound', 
'bluetick', 'border_collie', 'border_terrier', 'borzoi', 'boston_bulldog', 'bouvier_des_flandres', 'boxer', 
'brabancon_griffon', 'briard', 'brittany_spaniel', 'bull_mastiff', 'cairn', 'cardigan_corgi', 
'chesapeake_bay_retriever', 'chihuahua', 'chow', 'clumber', 'cocker_spaniel', 'collie', 'curly_coated_terrier', 
'dandie_dinmont', 'dhole', 'dingo', 'doberman', 'english_foxhound', 'english_setter', 'english_springer', 
'entlebucher', 'eskimo_dog', 'flat_coated_terrier', 'french_bulldog', 'german_shephard', 
'german_short_haired_pointer', 'giant_schnauzer', 'golden_retriever', 'gordon_setter', 'great_dane', 
'great_pyrenees', 'great_swiss_mountain_dog', 'groenendael', 'ibizan_hound', 'irish_setter', 'irish_terrier', 
'irish_water_spaniel', 'irish_wolfhound', 'italian_greyhound', 'japanese_spaniel', 'keeshond', 'kelpie', 
'kerry_blue_terrier', 'komondor', 'kuvasz', 'labrador_retriever', 'lakeland_terrier', 'leonberg', 'lhasa_apso', 
'malinois', 'maltese', 'mexican_hairless', 'miniature_pinscher', 'miniature_poodle', 'miniature_schnauzer', 
'newfoundland_terrier', 'norfolk_terrier', 'norwegian_elkhound', 'norwich_terrier', 'old_english_sheepdog', 
'otterhound', 'papillion', 'pekinese', 'pembroke_corgi', 'pomeranian', 'pug', 'redbone', 'rhodesian_ridgeback', 
'rottweiler', 'saint_bernard', 'saluki', 'samoyed', 'schipperke', 'scotch_terrier', 'scottish_deerhound', 
'sealyham_terrier', 'shetland_sheepdog', 'shih_tzu', 'siberian_husky', 'silky_terrier', 'soft_coated_wheaten_terrier', 
'staffordshire_terrier', 'standard_poodle', 'standard_schnauzer', 'sussex_spaniel', 'tibetan_mastiff', 
'tibetan_terrier', 'toy_poodle', 'toy_terrier', 'vizsla', 'walker_hound', 'weimaraner', 'welsh_springer_terrier', 
'west_highland_white_terrier', 'whippet', 'wirehaired_fox_terrier', 'yorkshire_terrier'];


var markers = [];

for(var i = 0; i < features.length; i++)  {
    // Push markers into an array

    new L.Marker([features[i].geometry.coordinates[1], features[i].geometry.coordinates[0]], {
        icon: L.divIcon({
            html: '<img src="static/img/' + doggos[i] + '.jpg" />',
            className: 'image-icon',
            iconSize: [52, 52]
        }),
        color: "#404040",
    }).bindPopup("<h5>" + features[i].properties.breed + "</h5>").addTo(myMap);
}
