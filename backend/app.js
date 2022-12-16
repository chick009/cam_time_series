const express = require('express');
const bodyParser = require('body-parser');
const mongoose = require('mongoose');

const placesRoutes = require('./routes/places-routes');
const usersRoutes = require('./routes/users-routes');
const adminRoutes = require('./routes/admin-routes');
const eventsRoutes = require('./routes/events-routes');
const HttpError = require('./models/http-error');

const app = express();

app.use(bodyParser.json());
/*edit*/
app.use(bodyParser.urlencoded())
/*edit*/
app.use((req, res, next) => {
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader(
    'Access-Control-Allow-Headers',
    'Origin, X-Requested-With, Content-Type, Accept, Authorization'
  );
  res.setHeader('Access-Control-Allow-Methods', 'GET, POST, PATCH, DELETE');

  next();
});

app.use('/api/places', placesRoutes);
app.use('/api/users', usersRoutes);
app.use('/api/admin', adminRoutes);
app.use('/api/events', eventsRoutes);
app.use((req, res, next) => {
  const error = new HttpError('Could noco find this route.', 404);
  throw error;
});

app.use((error, req, res, next) => {
  if (res.headerSent) {
    return next(error);
  }
  res.status(error.code || 500);
  res.json({ message: error.message || 'An unknown error occurred!' });
});

// const dbURL = "mongodb://127.0.0.1/test_db";

// mongoose
//   .connect(
//     dbURL,{ useNewUrlParser: true, useUnifiedTopology: true }
//   )
//   .then(() => {
//     app.listen(5000);
//   })
//   .catch(err => {
//     console.log(err);
//   });

mongoose
  .connect(
    `mongodb+srv://johnny5:Asd789156@cluster0.a2irl8c.mongodb.net/test`
  )
  .then(() => {
    app.listen(5000);
  })
  .catch(err => {
    console.log(err);
  });
