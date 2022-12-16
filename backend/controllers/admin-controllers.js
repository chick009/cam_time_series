const { validationResult } = require('express-validator');

const HttpError = require('../models/http-error');
const User = require('../models/user');
const Event = require('../models/event');
const Comment = require('../models/comment');
const Location = require('../models/location');
const getEvents = async (req, res, next) => {
    let events;
    try {
        events = await Event.find({},);
    } catch (err) {
      console.error("getEvents error:",err)
        const error = new HttpError(
            'getEvents failed, please try again later.',
            500
        );
        return next(error);
    }
    res.json({ events: events.map(event => event.toObject({ getters: true })) });
};

const getEvent = async (req, res, next) => {
    const { id } = req.params;
    let event;
    try {
        event = await Event.findOne({ _id:id },);
    } catch (err) {
      console.error("getEvent error:",err)
        const error = new HttpError(
            'getEvent failed, please try again later.',
            500
        );
        return next(error);
    }
    res.json({ event: event.toObject({ getters: true }) });
};

const getLocations = async (req, res, next) => {
    let locations;
    try {
        locations = await Location.find({},);
    } catch (err) {
        const error = new HttpError(
            'getLocations failed, please try again later.',
            500
        );
        return next(error);
    }
    res.json({ locations: locations.map(location => location.toObject({ getters: true })) });
};

const createEvent = async (req, res, next) => {
    const {
        title,
        venue,
        date,
        description,
        presenter,
        price
    } = req.body

    const createdEvent = new Event({
        title,
        venue,
        date,
        description,
        presenter,
        price,
      });
    
      try {
        await createdEvent.save();
      } catch (err) {
        const error = new HttpError(
          'create Event failed, please try again later.',
          500
        );
        return next(error);
      }
    
      res.status(201).json({ user: createdEvent.toObject({ getters: true }) });
}


const updateEvent = async (req, res, next) => {
  let doc = {};

  const { id } = req.params;
  const { ...updateField } = req.body;

  try {
    doc = await Event.findOneAndUpdate({_id:id},{...updateField}, {
      new: true
    });
  } catch (err) {
    console.error("updateEvent error:",err)
    const error = new HttpError(
      'update event failed, please try again later.',
      500
    );
    return next(error);
  }
  res.status(200).json({ doc:doc });
};

const deleteEvent = async (req, res, next) => {
    let result = {};
    const { id } = req.params;
    try {
        
        result = await Event.deleteOne({'_id':id}, {
        new: true
      });
    } catch (err) {
      console.log("deleteEvent",err)
      const error = new HttpError(
        'delete event failed, please try again later.',
        500
      );
      return next(error);
    }
    res.status(200).json({ result: result });
  };

exports.getEvents = getEvents;
exports.getEvent = getEvent;
exports.getLocations = getLocations;
exports.createEvent = createEvent;
exports.updateEvent = updateEvent;
exports.deleteEvent = deleteEvent;
