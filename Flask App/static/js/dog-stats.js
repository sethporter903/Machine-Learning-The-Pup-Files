// Read the data from CSV
d3.csv('https://raw.githubusercontent.com/krithikaragha/The-Pup-Files/master/Flask%20App/static/data/dogs.csv', function(data) {
    
    // Grab the location column values from the CSV file and add it to an array
    var locations = data.map(function(d) { return d.Location });
    console.log(locations);

    // Count each location and store it in a dictionary
    var locationCounts = locations.reduce((map, val) => {map[val] = (map[val] || 0); return map}, {} );
    locations.forEach(function(x) { locationCounts[x] = (locationCounts[x] || 0) + 1; });

    console.log(locationCounts);

    var pieData = [{
        values: Object.values(locationCounts),
        labels: Object.keys(locationCounts),
        type: 'pie'
    }];
      
    var layout = {
        height: 800,
        width: 1200
    };
      
    Plotly.newPlot('pie', pieData, layout);
    
});