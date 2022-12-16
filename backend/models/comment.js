const mongoose = require('mongoose');
const uniqueValidator = require('mongoose-unique-validator');

const Schema = mongoose.Schema;

const commentSchema = new Schema({
  username: { type: String, required: true },
  content: { type: String, required: true},
  date: { type: Date, required: true},
});

commentSchema.plugin(uniqueValidator);

module.exports = mongoose.model('comment', commentSchema);
