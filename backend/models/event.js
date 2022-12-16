const mongoose = require('mongoose');
// const AutoIncrement  = require('mongoose-sequence')(mongoose)
const uniqueValidator = require('mongoose-unique-validator');

const Schema = mongoose.Schema;

const eventSchema = new Schema({
  title: { type: String, required: true },
  venue: { type: String, required: true },
  date: { type: Date, required: true, minlength: 6 },
  description: { type: String, },
  presenter: { type: String},
  price: { type: Number, required: true}
});

// eventSchema.plugin(AutoIncrement, {inc_field: 'eventId'});
eventSchema.plugin(uniqueValidator);

module.exports = mongoose.model('event', eventSchema);
