const mongoose = require('mongoose');
const uniqueValidator = require('mongoose-unique-validator');

const Schema = mongoose.Schema;

const locationSchema = new Schema({
  venue: { type: String, required: true },
  latitude: { type: Number, required: true },
  longitude: { type: Number, required: true,},
  comment: [{ type: mongoose.Types.ObjectId, ref: 'Comment' }]
});

locationSchema.plugin(uniqueValidator);

module.exports = mongoose.model('location', locationSchema);
