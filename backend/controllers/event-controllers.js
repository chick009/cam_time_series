const HttpError = require('../models/http-error');
const Event = require('../models/event');
const axios = require('axios');
const xmljs = require('xml-js');
const moment = require('moment');
const xmlURL = 'https://www.lcsd.gov.hk/datagovhk/event/events.xml';

const venueURL = 'https://www.lcsd.gov.hk/datagovhk/event/venues.xml';
const dateURL = 'https://www.lcsd.gov.hk/datagovhk/event/eventDates.xml';
const fetchAllDataFromXMLAndUpdateDB = async (req, res, next) => {
    const response = await axios.get(
        xmlURL
    );
    const venueResponse = await axios.get(
        venueURL
    );
    const dateResponse = await axios.get(
        dateURL
    );
    const dataMapping = {
        titlee: 'title',
        venueid: 'venue',
        predateE: 'date',
        desce: 'description',
        presenterorge: 'presenter',
        pricee: 'price'
    }

    let venueMapping = {}
    let dateMapping = {}
    let eventArray = []
    const eventJson = JSON.parse(xmljs.xml2json(response.data))
    const venueJson = JSON.parse(xmljs.xml2json(venueResponse.data))
    const dateJson = JSON.parse(xmljs.xml2json(dateResponse.data))
    if (venueJson.elements[0].elements instanceof Array) {
        venueJson.elements[0].elements.forEach((venueObj, index) => {
            let tmpVenue = venueObj.elements.find(venue => {
                return venue.name === 'venuee'
            })
            venueMapping[venueObj.attributes.id] = tmpVenue.elements[0].cdata
        })
    }

    if (dateJson.elements[0].elements instanceof Array) {
        dateJson.elements[0].elements.forEach((dateObj, index) => {
            //assume put the first date in from xml
            let dateMoment = moment(dateObj.elements[0].elements[0].cdata,"YYYYMMDD")
            dateMapping[dateObj.attributes.id] = dateMoment.toDate()
        })
    }


    if (eventJson.elements[0].elements instanceof Array) {
        eventArray = eventJson.elements[0].elements.map((event, index) => {
            const elementList = event.elements;
            const desiredElements = elementList.filter((element) => Object.keys(dataMapping).includes(element.name) || element.name == 'eventid')
            index == 15 && console.log("desiredElements",desiredElements);
            const resultObject = desiredElements.reduce((result, element) => {
                const { name, elements, ...rest } = element;
                let key = dataMapping[name]
                switch(name){
                    case 'venueid':
                        return { ...result, [key]: elements && elements[0] ? venueMapping[elements[0].cdata] : '' }
                    case 'predateE':
                        return { ...result, [key]: elements && elements[0] ? dateMapping[event.attributes.id] : new Date('0001-01-01T00:00:00Z') }
                    case 'pricee':
                        let price = 0;
                        if(elements && elements[0]){
                            const regex = '[$]\\d+(?:,\\d+)?\\s*';
                            const found = elements[0].cdata.match(regex);
                            //use first found price by regex, if null is assume to be 0
                            if(found && found[0])
                                price = parseFloat(found[0].replace(/\D/g,''));
                        }    
                        return { ...result, [key]: price };
                    default:
                        return { ...result, [key]: elements && elements[0] ? elements[0].cdata : '' };
                }
            }, {});
            return resultObject
        })
    }
    // console.log("getEventsFromXML", eventJson.elements[0].elements[0], venueJson.elements[0].elements[0], venueMapping);
    // console.log("getEventsFromXML", eventArray);
    let dbResponse;
    try{
        dbResponse = await Event.insertMany(eventArray)
        console.log(eventArray.length,"items have been added!")
    }catch (err){
        console.error("getEventsFromXML error:",err)
        const error = new HttpError(
            'add event from xml failed, please try again later.',
            500
          );
          return next(error);
    }
    return res.status(201).json({ dbResponse: dbResponse.toObject({ getters: true }) });
    // eventArray.
};

exports.fetchAllDataFromXMLAndUpdateDB = fetchAllDataFromXMLAndUpdateDB